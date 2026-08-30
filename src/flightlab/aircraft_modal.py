from typing import NamedTuple

import numpy as np

from flightlab.state_space import ModalFamilyCharacterization


class AircraftModalFamilyCharacterization(NamedTuple):
    """Aircraft-state labels attached to one generic family characterization."""

    characterization: ModalFamilyCharacterization
    state_labels: tuple[str, ...]
    dominant_state_labels: tuple[str, ...]
    input_labels: tuple[str, ...] = ()
    dominant_input_labels: tuple[str, ...] = ()
    output_labels: tuple[str, ...] = ()
    dominant_output_labels: tuple[str, ...] = ()

    @property
    def state_participation_by_label(self):
        """Return participation values paired with labels in model state order."""
        values = self.characterization.state_participation.participation_magnitudes
        return tuple(zip(self.state_labels, values, strict=True))

    @property
    def input_influence_by_label(self):
        """Return input influences paired with labels in model input order."""
        values = self.characterization.input_influence.influence_magnitudes
        return tuple(zip(self.input_labels, values, strict=True))

    @property
    def output_influence_by_label(self):
        """Return output influences paired with labels in model output order."""
        values = self.characterization.output_influence.influence_magnitudes
        return tuple(zip(self.output_labels, values, strict=True))


def _validate_channel_mapping(values, dominant_indices, labels, channel):
    values = np.asarray(values)
    if values.ndim != 1 or values.shape != (len(labels),):
        raise ValueError(
            f"{channel} influence values must match the model {channel} dimension"
        )
    if any(
        not isinstance(index, (int, np.integer))
        or index < 0
        or index >= len(labels)
        for index in dominant_indices
    ):
        raise ValueError(
            f"dominant {channel} index is invalid for model {channel} labels"
        )
    return tuple(labels[index] for index in dominant_indices)


def interpret_modal_family_state_labels(
    characterizations, state_labels, input_labels=None, output_labels=None
):
    """Attach validated aircraft labels without changing generic results."""
    state_labels = tuple(state_labels)
    map_inputs = input_labels is not None
    map_outputs = output_labels is not None
    input_labels = () if input_labels is None else tuple(input_labels)
    output_labels = () if output_labels is None else tuple(output_labels)
    interpreted = []

    for characterization in characterizations:
        state_participation = characterization.state_participation
        values = np.asarray(state_participation.participation_magnitudes)
        if values.ndim != 1 or values.shape != (len(state_labels),):
            raise ValueError(
                "state participation values must match the model state dimension"
            )

        dominant_indices = state_participation.dominant_state_indices
        if any(
            not isinstance(index, (int, np.integer))
            or index < 0
            or index >= len(state_labels)
            for index in dominant_indices
        ):
            raise ValueError("dominant state index is invalid for model state labels")

        dominant_input_labels = ()
        if map_inputs:
            input_influence = characterization.input_influence
            dominant_input_labels = _validate_channel_mapping(
                input_influence.influence_magnitudes,
                input_influence.dominant_input_indices,
                input_labels,
                "input",
            )
        dominant_output_labels = ()
        if map_outputs:
            output_influence = characterization.output_influence
            dominant_output_labels = _validate_channel_mapping(
                output_influence.influence_magnitudes,
                output_influence.dominant_output_indices,
                output_labels,
                "output",
            )

        interpreted.append(
            AircraftModalFamilyCharacterization(
                characterization=characterization,
                state_labels=state_labels,
                dominant_state_labels=tuple(
                    state_labels[index] for index in dominant_indices
                ),
                input_labels=input_labels,
                dominant_input_labels=dominant_input_labels,
                output_labels=output_labels,
                dominant_output_labels=dominant_output_labels,
            )
        )

    return tuple(interpreted)


def filter_aircraft_modal_family_characterizations(
    characterizations,
    oscillatory=None,
    stability=None,
    dominant_state_labels=None,
    dominant_input_labels=None,
    dominant_output_labels=None,
    dominant_label_match="ANY",
    dominant_state_label_match=None,
    dominant_input_label_match=None,
    dominant_output_label_match=None,
    exclude_dominant_state_labels=None,
    exclude_dominant_input_labels=None,
    exclude_dominant_output_labels=None,
    exclude_dominant_label_match="ANY",
    exclude_dominant_state_label_match=None,
    exclude_dominant_input_label_match=None,
    exclude_dominant_output_label_match=None,
    state_labels=(),
    input_labels=(),
    output_labels=(),
):
    """Filter existing aircraft characterizations by exact categorical data."""
    if oscillatory is not None and not isinstance(oscillatory, bool):
        raise ValueError("oscillatory must be True, False, or None")
    valid_stabilities = {"decaying", "growing", "neutral"}
    if stability is not None and stability not in valid_stabilities:
        raise ValueError(
            "stability must be 'decaying', 'growing', 'neutral', or None"
        )
    valid_label_matches = {"ANY", "ALL", "EXACT"}
    if dominant_label_match not in valid_label_matches:
        raise ValueError("dominant_label_match must be 'ANY', 'ALL', or 'EXACT'")
    category_label_matches = {
        "state": dominant_state_label_match,
        "input": dominant_input_label_match,
        "output": dominant_output_label_match,
    }
    for category, match in category_label_matches.items():
        if match is not None and match not in valid_label_matches:
            raise ValueError(
                f"dominant_{category}_label_match must be 'ANY', 'ALL', "
                "'EXACT', or None"
            )
        if match is None:
            category_label_matches[category] = dominant_label_match
    if exclude_dominant_label_match not in valid_label_matches:
        raise ValueError(
            "exclude_dominant_label_match must be 'ANY', 'ALL', or 'EXACT'"
        )
    category_exclusion_matches = {
        "state": exclude_dominant_state_label_match,
        "input": exclude_dominant_input_label_match,
        "output": exclude_dominant_output_label_match,
    }
    for category, match in category_exclusion_matches.items():
        if match is not None and match not in valid_label_matches:
            raise ValueError(
                f"exclude_dominant_{category}_label_match must be 'ANY', 'ALL', "
                "'EXACT', or None"
            )
        if match is None:
            category_exclusion_matches[category] = exclude_dominant_label_match

    def normalized_label_filter(requested, valid_labels, category, exclude=False):
        if requested is None:
            return None
        filter_name = f"{'excluded ' if exclude else ''}dominant {category} label"
        requested_labels = (requested,) if isinstance(requested, str) else tuple(requested)
        if not requested_labels:
            raise ValueError(f"{filter_name} filter must not be empty")
        invalid_labels = tuple(
            label for label in requested_labels if label not in valid_labels
        )
        if invalid_labels:
            raise ValueError(f"invalid {filter_name}: {invalid_labels[0]!r}")
        return frozenset(requested_labels)

    requested_states = normalized_label_filter(
        dominant_state_labels, state_labels, "state"
    )
    requested_inputs = normalized_label_filter(
        dominant_input_labels, input_labels, "input"
    )
    requested_outputs = normalized_label_filter(
        dominant_output_labels, output_labels, "output"
    )
    excluded_states = normalized_label_filter(
        exclude_dominant_state_labels, state_labels, "state", exclude=True
    )
    excluded_inputs = normalized_label_filter(
        exclude_dominant_input_labels, input_labels, "input", exclude=True
    )
    excluded_outputs = normalized_label_filter(
        exclude_dominant_output_labels, output_labels, "output", exclude=True
    )

    def labels_match(requested, dominant, match):
        if requested is None:
            return True
        dominant = frozenset(dominant)
        if match == "ANY":
            return not requested.isdisjoint(dominant)
        if match == "ALL":
            return requested <= dominant
        return requested == dominant

    return tuple(
        characterization
        for characterization in characterizations
        if (
            oscillatory is None
            or characterization.characterization.dynamics.is_oscillatory
            is oscillatory
        )
        and (
            stability is None
            or characterization.characterization.dynamics.stability == stability
        )
        and labels_match(
            requested_states,
            characterization.dominant_state_labels,
            category_label_matches["state"],
        )
        and labels_match(
            requested_inputs,
            characterization.dominant_input_labels,
            category_label_matches["input"],
        )
        and labels_match(
            requested_outputs,
            characterization.dominant_output_labels,
            category_label_matches["output"],
        )
        and (
            excluded_states is None
            or not labels_match(
                excluded_states,
                characterization.dominant_state_labels,
                category_exclusion_matches["state"],
            )
        )
        and (
            excluded_inputs is None
            or not labels_match(
                excluded_inputs,
                characterization.dominant_input_labels,
                category_exclusion_matches["input"],
            )
        )
        and (
            excluded_outputs is None
            or not labels_match(
                excluded_outputs,
                characterization.dominant_output_labels,
                category_exclusion_matches["output"],
            )
        )
    )
