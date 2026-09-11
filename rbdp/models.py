"""Mask-aware autoencoders and uncertainty-aware dual predictors.

The architecture follows the lightweight MLP/dual-prediction design used by
COMPLETER and DCP, while exposing every tensor needed by the proposed risk and
reliability analyses.  It does not import or modify the legacy implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm that uses running statistics for a one-sample masked subset."""

    def forward(self, values):
        if self.training and values.size(0) <= 1:
            return F.batch_norm(
                values,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )
        return super(SafeBatchNorm1d, self).forward(values)


def _hidden_block(input_dim, output_dim, batch_norm):
    layers = [nn.Linear(int(input_dim), int(output_dim))]
    if batch_norm:
        layers.append(SafeBatchNorm1d(int(output_dim)))
    layers.append(nn.ReLU(inplace=True))
    return layers


class ViewAutoencoder(nn.Module):
    """Encode one view into a categorical latent vector and reconstruct it.

    Input ``x`` is ``FloatTensor [B_v, D_v]`` containing only observed rows.
    ``encode`` returns ``FloatTensor [B_v, d]`` on the probability simplex;
    ``decode`` maps it back to ``FloatTensor [B_v, D_v]``.  Missing rows are
    filtered by :class:`RBDPModel` before this module is called.
    """

    def __init__(self, input_dim, hidden_dims, latent_dim, batch_norm=True):
        super(ViewAutoencoder, self).__init__()
        input_dim = int(input_dim)
        latent_dim = int(latent_dim)
        hidden_dims = [int(value) for value in hidden_dims]
        if input_dim <= 0 or latent_dim <= 1 or any(value <= 0 for value in hidden_dims):
            raise ValueError("autoencoder dimensions must be positive and latent_dim > 1")

        encoder_layers = []
        previous = input_dim
        for hidden in hidden_dims:
            encoder_layers.extend(_hidden_block(previous, hidden, batch_norm))
            previous = hidden
        encoder_layers.append(nn.Linear(previous, latent_dim))
        self.encoder_network = nn.Sequential(*encoder_layers)

        decoder_layers = []
        previous = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.extend(_hidden_block(previous, hidden, batch_norm))
            previous = hidden
        decoder_layers.append(nn.Linear(previous, input_dim))
        self.decoder_network = nn.Sequential(*decoder_layers)

    def encode(self, values):
        return F.softmax(self.encoder_network(values), dim=1)

    def decode(self, latent):
        return self.decoder_network(latent)

    def forward(self, values):
        latent = self.encode(values)
        return self.decode(latent), latent


class UncertainPredictor(nn.Module):
    """Predict a target latent and a scalar log variance from a source latent.

    Input is ``FloatTensor [B_st, d]``.  Outputs are ``mean [B_st, d]`` on the
    simplex and ``log_variance [B_st, 1]``.  A scalar uncertainty adds only one
    output per sample and is more stable than a free variance for every latent
    coordinate.  The initial log variance is -4 and is clamped during use.
    """

    def __init__(self, latent_dim, hidden_dims, batch_norm=True):
        super(UncertainPredictor, self).__init__()
        latent_dim = int(latent_dim)
        hidden_dims = [int(value) for value in hidden_dims]
        layers = []
        previous = latent_dim
        for hidden in hidden_dims:
            layers.extend(_hidden_block(previous, hidden, batch_norm))
            previous = hidden
        self.trunk = nn.Sequential(*layers) if layers else None
        self.mean_head = nn.Linear(previous, latent_dim)
        self.log_variance_head = nn.Linear(previous, 1)
        nn.init.zeros_(self.log_variance_head.weight)
        nn.init.constant_(self.log_variance_head.bias, -4.0)

    def forward(self, source_latent):
        hidden = source_latent if self.trunk is None else self.trunk(source_latent)
        mean = F.softmax(self.mean_head(hidden), dim=1)
        log_variance = self.log_variance_head(hidden)
        return mean, log_variance


class RBDPModel(nn.Module):
    """Multi-view autoencoding plus all directed latent predictors.

    Constructor arguments ``view_dims`` and ``hidden_dims`` describe the input
    feature dimensions and per-view encoder hidden layers.  Every view shares a
    common latent dimension ``d`` but has independent parameters.  Predictors
    are distinct for all ordered pairs ``s -> t``.
    """

    def __init__(self, view_dims, hidden_dims, latent_dim,
                 predictor_hidden_dims, batch_norm=True,
                 min_log_variance=-8.0, max_log_variance=3.0):
        super(RBDPModel, self).__init__()
        self.view_dims = [int(value) for value in view_dims]
        self.n_views = len(self.view_dims)
        self.latent_dim = int(latent_dim)
        if self.n_views < 2:
            raise ValueError("RBDPModel needs at least two views")
        if len(hidden_dims) != self.n_views:
            raise ValueError("hidden_dims must contain one list per view")
        self.min_log_variance = float(min_log_variance)
        self.max_log_variance = float(max_log_variance)
        if self.min_log_variance >= self.max_log_variance:
            raise ValueError("min_log_variance must be smaller than max_log_variance")

        self.autoencoders = nn.ModuleList([
            ViewAutoencoder(dimension, hidden, self.latent_dim, batch_norm)
            for dimension, hidden in zip(self.view_dims, hidden_dims)
        ])
        predictors = {}
        for source in range(self.n_views):
            for target in range(self.n_views):
                if source != target:
                    predictors[self.predictor_key(source, target)] = UncertainPredictor(
                        self.latent_dim, predictor_hidden_dims, batch_norm)
        self.predictors = nn.ModuleDict(predictors)

    @staticmethod
    def predictor_key(source, target):
        return "v%d_to_v%d" % (int(source), int(target))

    def _validate_inputs(self, views, mask):
        if len(views) != self.n_views:
            raise ValueError("expected %d views, received %d" %
                             (self.n_views, len(views)))
        if mask.dim() != 2 or mask.size(1) != self.n_views:
            raise ValueError("mask must have shape [B, %d]" % self.n_views)
        batch_size = mask.size(0)
        for index, (view, dimension) in enumerate(zip(views, self.view_dims)):
            if view.dim() != 2 or view.size(0) != batch_size or view.size(1) != dimension:
                raise ValueError("view %d must have shape [B, %d]" %
                                 (index, dimension))
            if view.device != mask.device:
                raise ValueError("views and mask must reside on the same device")
        if torch.any(mask.sum(dim=1) == 0).item():
            raise ValueError("every sample must retain at least one observed view")

    def encode_observed(self, views, mask):
        """Encode only rows marked observed.

        Args:
            views: Length-``V`` list of ``FloatTensor [B, D_v]``.  Values at
                missing rows may be arbitrary because they are never indexed.
            mask: Binary/boolean tensor ``[B, V]``; one means observed.

        Returns:
            Length-``V`` list of ``FloatTensor [B, d]``.  Missing positions are
            exact zeros and must not be interpreted as predictions.  Gradients
            flow only through observed rows.
        """
        self._validate_inputs(views, mask)
        latents = []
        for index, autoencoder in enumerate(self.autoencoders):
            observed = mask[:, index] > 0
            latent = views[index].new_zeros((views[index].size(0), self.latent_dim))
            if torch.any(observed).item():
                latent[observed] = autoencoder.encode(views[index][observed])
            latents.append(latent)
        return latents

    def encode_all(self, views):
        """Encode complete benchmark features for diagnostic metrics only."""
        if len(views) != self.n_views:
            raise ValueError("expected %d views" % self.n_views)
        return [autoencoder.encode(view) for autoencoder, view in
                zip(self.autoencoders, views)]

    def predict(self, source, target, source_latent):
        """Return ``(mean [B,d], clipped_log_variance [B,1])`` for ``s -> t``."""
        source = int(source)
        target = int(target)
        if source == target:
            raise ValueError("source and target views must differ")
        mean, log_variance = self.predictors[
            self.predictor_key(source, target)](source_latent)
        return mean, torch.clamp(
            log_variance, self.min_log_variance, self.max_log_variance)

    def complete(self, views, mask, use_reliability=True, gamma=1.0,
                 source_temperature=1.0, confidence_temperature=1.0,
                 risk_normalization="none",
                 fusion="consensus"):
        """Complete missing latents and expose reliability intermediates.

        Args:
            views: Observed/masked feature list, each ``FloatTensor [B,D_v]``.
            mask: Binary tensor ``[B,V]`` with at least one one per row.
            use_reliability: If false, available predictors receive equal
                source weights and all completed views receive confidence one.
            gamma: Weight of the cycle residual in path risk.
            source_temperature: Softmax temperature across source paths.
            confidence_temperature: Exponential gate temperature.
            risk_normalization: ``"none"`` uses raw risk units;
                ``"per_sample_range"`` maps the best/worst valid source risk
                to zero/one for each sample and target before softmax.  The
                latter makes temperature dimensionless across datasets.
            fusion: ``"consensus"`` returns ``[B,d]``; ``"concat"`` returns
                the DCP-compatible ``[B,V*d]`` completed concatenation; and
                ``"gated_concat"`` preserves that dimension while scaling
                predicted blocks. ``"fallback_concat"`` instead interpolates
                risky predictions toward the observed-view consensus, keeping
                every block on the same latent simplex scale.

        Returns:
            Dict containing ``representation``, ``observed_latents`` and
            ``completed_latents`` plus ``view_confidence [B,V]`` and path-level
            ``risk/log_variance/cycle/weight/valid [B,V,V]``.  Path dimensions
            are ordered ``[sample, source, target]``; invalid entries are -1
            for numeric diagnostics and false for ``path_valid``.
        """
        self._validate_inputs(views, mask)
        if gamma < 0 or source_temperature <= 0 or confidence_temperature <= 0:
            raise ValueError("reliability weights/temperatures must be positive")
        if risk_normalization not in ("none", "per_sample_range"):
            raise ValueError("unsupported risk normalization: %s" %
                             risk_normalization)
        if fusion not in ("consensus", "concat", "gated_concat", "fallback_concat"):
            raise ValueError("unsupported fusion mode: %s" % fusion)

        observed_latents = self.encode_observed(views, mask)
        batch_size = mask.size(0)
        completed = torch.stack(observed_latents, dim=1)
        view_confidence = completed.new_ones((batch_size, self.n_views))
        path_risk = completed.new_full((batch_size, self.n_views, self.n_views), -1.0)
        path_log_variance = path_risk.clone()
        path_cycle = path_risk.clone()
        path_weight = completed.new_zeros((batch_size, self.n_views, self.n_views))
        path_valid = mask.new_zeros(
            (batch_size, self.n_views, self.n_views), dtype=torch.uint8)

        for target in range(self.n_views):
            missing_target = mask[:, target] == 0
            if not torch.any(missing_target).item():
                continue
            means = []
            risks = []
            valids = []
            for source in range(self.n_views):
                mean_buffer = completed.new_zeros((batch_size, self.latent_dim))
                risk_buffer = completed.new_full((batch_size,), 1.0e9)
                valid = missing_target & (mask[:, source] > 0) if source != target else (
                    missing_target & (mask[:, source] > 1))
                if source != target and torch.any(valid).item():
                    mean, log_variance = self.predict(
                        source, target, observed_latents[source][valid])
                    returned, _ = self.predict(target, source, mean)
                    cycle = torch.mean(
                        (returned - observed_latents[source][valid]) ** 2, dim=1)
                    risk = torch.exp(log_variance.reshape(-1)) + float(gamma) * cycle
                    mean_buffer[valid] = mean
                    risk_buffer[valid] = risk
                    path_risk[valid, source, target] = risk
                    path_log_variance[valid, source, target] = log_variance.reshape(-1)
                    path_cycle[valid, source, target] = cycle
                    path_valid[valid, source, target] = 1
                means.append(mean_buffer)
                risks.append(risk_buffer)
                valids.append(valid)

            means_tensor = torch.stack(means, dim=1)
            risks_tensor = torch.stack(risks, dim=1)
            valid_tensor = torch.stack(valids, dim=1)
            valid_float = valid_tensor.float()
            if use_reliability:
                weighting_risk = risks_tensor
                if risk_normalization == "per_sample_range":
                    lower = risks_tensor.masked_fill(
                        ~valid_tensor, 1.0e9).min(dim=1, keepdim=True)[0]
                    upper = risks_tensor.masked_fill(
                        ~valid_tensor, -1.0e9).max(dim=1, keepdim=True)[0]
                    scale = torch.clamp(upper - lower, min=1.0e-8)
                    weighting_risk = (risks_tensor - lower) / scale
                logits = -weighting_risk / float(source_temperature)
                logits = logits.masked_fill(~valid_tensor, -1.0e9)
                source_weights = F.softmax(logits, dim=1) * valid_float
                source_weights = source_weights / torch.clamp(
                    source_weights.sum(dim=1, keepdim=True), min=1.0e-12)
            else:
                source_weights = valid_float / torch.clamp(
                    valid_float.sum(dim=1, keepdim=True), min=1.0)
            predicted_target = torch.sum(
                source_weights.unsqueeze(2) * means_tensor, dim=1)
            completed[missing_target, target] = predicted_target[missing_target]
            weighted_risk = torch.sum(
                source_weights * torch.where(
                    valid_tensor, risks_tensor, risks_tensor.new_zeros(risks_tensor.shape)),
                dim=1,
            )
            if use_reliability:
                confidence = torch.exp(
                    -weighted_risk / float(confidence_temperature)).clamp(0.0, 1.0)
                view_confidence[missing_target, target] = confidence[missing_target]
            for source in range(self.n_views):
                path_weight[:, source, target] = source_weights[:, source]

        concat = completed.reshape(batch_size, self.n_views * self.latent_dim)
        weights = torch.where(mask > 0, view_confidence.new_ones(mask.shape),
                              view_confidence)
        gated_concat = (completed * weights.unsqueeze(2)).reshape(
            batch_size, self.n_views * self.latent_dim)
        observed_weights = (mask > 0).float()
        observed_consensus = torch.sum(
            torch.stack(observed_latents, dim=1), dim=1) / torch.clamp(
                observed_weights.sum(dim=1, keepdim=True), min=1.0)
        fallback_latents = completed.clone()
        for target in range(self.n_views):
            missing_target = mask[:, target] == 0
            if torch.any(missing_target).item():
                confidence = view_confidence[missing_target, target:target + 1]
                fallback_latents[missing_target, target] = (
                    confidence * completed[missing_target, target] +
                    (1.0 - confidence) * observed_consensus[missing_target])
        fallback_concat = fallback_latents.reshape(
            batch_size, self.n_views * self.latent_dim)
        consensus = torch.sum(completed * weights.unsqueeze(2), dim=1) / torch.clamp(
            weights.sum(dim=1, keepdim=True), min=1.0e-12)
        if fusion == "consensus":
            representation = consensus
        elif fusion == "fallback_concat":
            representation = fallback_concat
        elif fusion == "gated_concat":
            representation = gated_concat
        else:
            representation = concat
        return {
            "representation": representation,
            "concat_representation": concat,
            "gated_concat_representation": gated_concat,
            "fallback_concat_representation": fallback_concat,
            "consensus_representation": consensus,
            "observed_latents": torch.stack(observed_latents, dim=1),
            "completed_latents": completed,
            "view_confidence": view_confidence,
            "path_risk": path_risk,
            "path_log_variance": path_log_variance,
            "path_cycle": path_cycle,
            "path_weight": path_weight,
            "path_valid": path_valid,
        }
