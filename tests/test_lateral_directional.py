from types import SimpleNamespace

import numpy as np
import pytest

from flightlab.lateral_directional import LateralDirectionalModel
from flightlab.state_space import StateSpace


def valid_parameters():
    return {
        "trim_speed": 10,
        "trim_pitch": 0,
        "gravity": 9,
        "y_v": 1,
        "y_p": 2,
        "y_r": 3,
        "y_delta_a": 4,
        "y_delta_r": 5,
        "l_v": 6,
        "l_p": 7,
        "l_r": 8,
        "l_delta_a": 9,
        "l_delta_r": 10,
        "n_v": 11,
        "n_p": 12,
        "n_r": 13,
        "n_delta_a": 14,
        "n_delta_r": 15,
    }


def generic_characterization(
    dominant_indices,
    values=None,
    input_values=(0.6, 0.4),
    dominant_input_indices=(0,),
    output_values=(0.1, 0.2, 0.3, 0.4),
    dominant_output_indices=(3,),
):
    if values is None:
        values = np.full(4, 0.1)
    return SimpleNamespace(
        dynamics=object(),
        state_participation=SimpleNamespace(
            participation_magnitudes=np.asarray(values),
            dominant_state_indices=dominant_indices,
        ),
        input_influence=SimpleNamespace(
            influence_magnitudes=np.asarray(input_values),
            dominant_input_indices=dominant_input_indices,
        ),
        output_influence=SimpleNamespace(
            influence_magnitudes=np.asarray(output_values),
            dominant_output_indices=dominant_output_indices,
        ),
    )


class FakeStateSpace:
    def __init__(self, characterizations):
        self.characterizations = characterizations

    def modal_family_characterizations(self):
        return self.characterizations


def filtered_characterization(
    oscillatory,
    stability,
    states=(),
    inputs=(),
    outputs=(),
):
    return SimpleNamespace(
        characterization=SimpleNamespace(
            dynamics=SimpleNamespace(
                is_oscillatory=oscillatory, stability=stability
            )
        ),
        dominant_state_labels=states,
        dominant_input_labels=inputs,
        dominant_output_labels=outputs,
    )


def test_lateral_directional_model_builds_expected_state_space_matrices():
    model = LateralDirectionalModel(**valid_parameters())

    system = model.to_state_space()

    assert isinstance(system, StateSpace)
    np.testing.assert_array_equal(
        system.A,
        [[1, 2, -7, 9], [6, 7, 8, 0], [11, 12, 13, 0], [0, 1, 0, 0]],
    )
    np.testing.assert_array_equal(
        system.B, [[4, 5], [9, 10], [14, 15], [0, 0]]
    )
    np.testing.assert_array_equal(system.C, np.eye(4))
    np.testing.assert_array_equal(system.D, np.zeros((4, 2)))
    assert system.eigenvalues().shape == (4,)
    modes = system.modal_properties()
    assert len(modes) == 4
    np.testing.assert_allclose([mode.eigenvalue for mode in modes], system.eigenvalues())
    eigenvalues = system.eigenvalues()
    eigenvectors = system.right_eigenvectors()
    left_eigenvectors = system.left_eigenvectors()
    biorthogonal_modes = system.biorthogonal_modes()
    participation = system.participation_factors()
    modal_input = system.modal_input_influence()
    modal_output = system.modal_output_influence()
    assert eigenvectors.shape == (4, 4)
    assert left_eigenvectors.shape == (4, 4)
    assert biorthogonal_modes.right_eigenvectors.shape == (4, 4)
    assert biorthogonal_modes.left_eigenvectors.shape == (4, 4)
    assert participation.shape == (4, 4)
    assert modal_input.shape == (4, 2)
    assert modal_output.shape == (4, 4)
    assert np.all(np.isfinite(eigenvalues))
    assert np.all(np.isfinite(eigenvectors.real))
    assert np.all(np.isfinite(eigenvectors.imag))
    assert np.all(np.isfinite(left_eigenvectors.real))
    assert np.all(np.isfinite(left_eigenvectors.imag))
    assert np.all(np.isfinite(biorthogonal_modes.right_eigenvectors.real))
    assert np.all(np.isfinite(biorthogonal_modes.right_eigenvectors.imag))
    assert np.all(np.isfinite(biorthogonal_modes.left_eigenvectors.real))
    assert np.all(np.isfinite(biorthogonal_modes.left_eigenvectors.imag))
    assert np.all(np.isfinite(participation.real))
    assert np.all(np.isfinite(participation.imag))
    assert np.all(np.isfinite(modal_input.real))
    assert np.all(np.isfinite(modal_input.imag))
    assert np.all(np.isfinite(modal_output.real))
    assert np.all(np.isfinite(modal_output.imag))
    np.testing.assert_array_equal(biorthogonal_modes.eigenvalues, eigenvalues)
    np.testing.assert_allclose(
        participation,
        biorthogonal_modes.right_eigenvectors
        * np.conj(biorthogonal_modes.left_eigenvectors),
    )
    np.testing.assert_allclose(np.sum(participation, axis=0), np.ones(4))
    np.testing.assert_allclose(
        modal_input, biorthogonal_modes.left_eigenvectors.conj().T @ system.B
    )
    np.testing.assert_allclose(
        modal_output, system.C @ biorthogonal_modes.right_eigenvectors
    )
    assert LateralDirectionalModel.INPUT_ORDER == ("delta_a", "delta_r")
    assert LateralDirectionalModel.OUTPUT_ORDER == LateralDirectionalModel.STATE_ORDER
    for index, eigenvalue in enumerate(eigenvalues):
        vector = eigenvectors[:, index]
        np.testing.assert_allclose(system.A @ vector, eigenvalue * vector)
        left_vector = left_eigenvectors[:, index]
        np.testing.assert_allclose(
            left_vector.conj().T @ system.A,
            eigenvalue * left_vector.conj().T,
        )
        scaled_right = biorthogonal_modes.right_eigenvectors[:, index]
        scaled_left = biorthogonal_modes.left_eigenvectors[:, index]
        np.testing.assert_allclose(scaled_left.conj().T @ scaled_right, 1.0)
        np.testing.assert_allclose(
            system.A @ scaled_right, eigenvalue * scaled_right
        )
        np.testing.assert_allclose(
            scaled_left.conj().T @ system.A,
            eigenvalue * scaled_left.conj().T,
        )
    assert isinstance(system.is_asymptotically_stable(), bool)
    assert system.rk4_step(np.zeros(4), np.zeros(2), 0.01).shape == (4,)


def test_lateral_directional_model_declares_state_input_and_output_ordering():
    assert LateralDirectionalModel.STATE_ORDER == ("v", "p", "r", "phi")
    assert LateralDirectionalModel.INPUT_ORDER == ("delta_a", "delta_r")
    assert LateralDirectionalModel.OUTPUT_ORDER == LateralDirectionalModel.STATE_ORDER


@pytest.mark.parametrize(
    ("index", "label"), tuple(enumerate(LateralDirectionalModel.STATE_ORDER))
)
def test_lateral_modal_family_characterization_maps_dominant_state(
    monkeypatch, index, label
):
    generic = generic_characterization((index,))
    monkeypatch.setattr(
        LateralDirectionalModel,
        "to_state_space",
        lambda self: FakeStateSpace((generic,)),
    )

    (interpreted,) = LateralDirectionalModel(
        **valid_parameters()
    ).modal_family_characterizations()

    assert interpreted.characterization is generic
    assert interpreted.state_labels == ("v", "p", "r", "phi")
    assert interpreted.dominant_state_labels == (label,)
    assert interpreted.characterization.dynamics is generic.dynamics
    assert interpreted.characterization.input_influence is generic.input_influence
    assert interpreted.characterization.output_influence is generic.output_influence


def test_lateral_modal_family_characterization_maps_tied_inputs_and_outputs(
    monkeypatch,
):
    generic = generic_characterization(
        (0,),
        input_values=(0.7, 0.7),
        dominant_input_indices=(0, 1),
        output_values=(0.9, 0.2, 0.9, 0.1),
        dominant_output_indices=(0, 2),
    )
    monkeypatch.setattr(
        LateralDirectionalModel,
        "to_state_space",
        lambda self: FakeStateSpace((generic,)),
    )

    (interpreted,) = LateralDirectionalModel(
        **valid_parameters()
    ).modal_family_characterizations()

    assert interpreted.input_labels == LateralDirectionalModel.INPUT_ORDER
    assert interpreted.dominant_input_labels == ("delta_a", "delta_r")
    assert interpreted.input_influence_by_label == (
        ("delta_a", 0.7),
        ("delta_r", 0.7),
    )
    assert interpreted.output_labels == LateralDirectionalModel.OUTPUT_ORDER
    assert interpreted.dominant_output_labels == ("v", "r")
    assert interpreted.output_influence_by_label == (
        ("v", 0.9),
        ("p", 0.2),
        ("r", 0.9),
        ("phi", 0.1),
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"input_values": (1.0,)}, "must match the model input dimension"),
        ({"output_values": (1.0, 2.0)}, "must match the model output dimension"),
        ({"dominant_input_indices": (2,)}, "dominant input index is invalid"),
        ({"dominant_output_indices": (4,)}, "dominant output index is invalid"),
    ],
)
def test_lateral_modal_family_characterization_validates_channel_mapping(
    monkeypatch, overrides, message
):
    generic = generic_characterization((0,), **overrides)
    monkeypatch.setattr(
        LateralDirectionalModel,
        "to_state_space",
        lambda self: FakeStateSpace((generic,)),
    )

    with pytest.raises(ValueError, match=message):
        LateralDirectionalModel(**valid_parameters()).modal_family_characterizations()


def test_lateral_modal_family_characterization_preserves_tie_and_order(monkeypatch):
    first = generic_characterization((0, 2, 3), [0.4, 0.1, 0.4, 0.4])
    second = generic_characterization((1,), [0.1, 0.7, 0.1, 0.1])
    monkeypatch.setattr(
        LateralDirectionalModel,
        "to_state_space",
        lambda self: FakeStateSpace((first, second)),
    )

    interpreted = LateralDirectionalModel(
        **valid_parameters()
    ).modal_family_characterizations()

    assert [result.characterization for result in interpreted] == [first, second]
    assert interpreted[0].dominant_state_labels == ("v", "r", "phi")
    assert interpreted[1].dominant_state_labels == ("p",)
    assert interpreted[0].state_participation_by_label == (
        ("v", 0.4),
        ("p", 0.1),
        ("r", 0.4),
        ("phi", 0.4),
    )


@pytest.mark.parametrize(
    ("indices", "values", "message"),
    [
        ((4,), np.full(4, 0.25), "dominant state index is invalid"),
        ((0,), np.full(3, 1 / 3), "must match the model state dimension"),
    ],
)
def test_lateral_modal_family_characterization_validates_state_mapping(
    monkeypatch, indices, values, message
):
    generic = generic_characterization(indices, values)
    monkeypatch.setattr(
        LateralDirectionalModel,
        "to_state_space",
        lambda self: FakeStateSpace((generic,)),
    )

    with pytest.raises(ValueError, match=message):
        LateralDirectionalModel(**valid_parameters()).modal_family_characterizations()


def test_lateral_filters_modal_family_characterizations_without_reordering(
    monkeypatch,
):
    characterizations = (
        filtered_characterization(True, "growing"),
        filtered_characterization(False, "decaying"),
        filtered_characterization(True, "decaying"),
        filtered_characterization(True, "neutral"),
    )
    monkeypatch.setattr(
        LateralDirectionalModel,
        "modal_family_characterizations",
        lambda self: characterizations,
    )

    model = LateralDirectionalModel(**valid_parameters())
    combined = model.filter_modal_family_characterizations(
        oscillatory=True, stability="decaying"
    )
    unfiltered = model.filter_modal_family_characterizations()

    assert combined == (characterizations[2],)
    assert combined[0] is characterizations[2]
    assert unfiltered == characterizations
    assert all(
        actual is expected
        for actual, expected in zip(unfiltered, characterizations, strict=True)
    )


@pytest.mark.parametrize("label", LateralDirectionalModel.STATE_ORDER)
def test_lateral_filter_matches_each_dominant_state_label(monkeypatch, label):
    characterizations = tuple(
        filtered_characterization(False, "decaying", states=(state_label,))
        for state_label in LateralDirectionalModel.STATE_ORDER
    )
    monkeypatch.setattr(
        LateralDirectionalModel,
        "modal_family_characterizations",
        lambda self: characterizations,
    )

    result = LateralDirectionalModel(
        **valid_parameters()
    ).filter_modal_family_characterizations(dominant_state_labels=label)

    expected = characterizations[LateralDirectionalModel.STATE_ORDER.index(label)]
    assert result == (expected,)
    assert result[0] is expected


def test_lateral_filter_uses_any_input_output_labels_and_combines(monkeypatch):
    characterizations = (
        filtered_characterization(
            True,
            "decaying",
            states=("v", "r"),
            inputs=("delta_a",),
            outputs=("v",),
        ),
        filtered_characterization(
            True,
            "decaying",
            states=("p",),
            inputs=("delta_r",),
            outputs=("p", "phi"),
        ),
        filtered_characterization(
            False,
            "neutral",
            states=("phi",),
            inputs=(),
            outputs=(),
        ),
    )
    monkeypatch.setattr(
        LateralDirectionalModel,
        "modal_family_characterizations",
        lambda self: characterizations,
    )
    model = LateralDirectionalModel(**valid_parameters())

    inputs = model.filter_modal_family_characterizations(
        dominant_input_labels=("delta_a", "delta_r")
    )
    outputs = model.filter_modal_family_characterizations(
        dominant_output_labels=("r", "phi")
    )
    combined = model.filter_modal_family_characterizations(
        oscillatory=True,
        stability="decaying",
        dominant_state_labels=("p", "r"),
        dominant_input_labels="delta_r",
        dominant_output_labels="phi",
    )

    assert inputs == characterizations[:2]
    assert outputs == (characterizations[1],)
    assert combined == (characterizations[1],)
    assert combined[0] is characterizations[1]


def test_lateral_filter_applies_exact_matching_to_each_label_category(monkeypatch):
    characterizations = (
        filtered_characterization(
            True,
            "decaying",
            states=("v", "r"),
            inputs=("delta_a", "delta_r"),
            outputs=("v", "r"),
        ),
        filtered_characterization(
            True,
            "decaying",
            states=("v", "r", "phi"),
            inputs=("delta_a", "delta_r"),
            outputs=("v", "r"),
        ),
    )
    monkeypatch.setattr(
        LateralDirectionalModel,
        "modal_family_characterizations",
        lambda self: characterizations,
    )

    result = LateralDirectionalModel(
        **valid_parameters()
    ).filter_modal_family_characterizations(
        dominant_state_labels=("r", "v"),
        dominant_input_labels=("delta_r", "delta_a"),
        dominant_output_labels=("r", "v"),
        dominant_label_match="EXACT",
    )

    assert result == (characterizations[0],)
    assert result[0] is characterizations[0]


def test_lateral_filter_overrides_match_mode_by_label_category(monkeypatch):
    characterizations = (
        filtered_characterization(
            True,
            "decaying",
            states=("v", "r"),
            inputs=("delta_a", "delta_r"),
            outputs=("v", "r"),
        ),
        filtered_characterization(
            True,
            "decaying",
            states=("v", "r", "phi"),
            inputs=("delta_a", "delta_r"),
            outputs=("v", "r", "phi"),
        ),
        filtered_characterization(
            False,
            "decaying",
            states=("v", "r"),
            inputs=("delta_a",),
            outputs=("v", "r"),
        ),
    )
    monkeypatch.setattr(
        LateralDirectionalModel,
        "modal_family_characterizations",
        lambda self: characterizations,
    )

    result = LateralDirectionalModel(
        **valid_parameters()
    ).filter_modal_family_characterizations(
        oscillatory=True,
        stability="decaying",
        dominant_state_labels=("v", "r", "r"),
        dominant_input_labels="delta_a",
        dominant_output_labels="v",
        dominant_label_match="EXACT",
        dominant_input_label_match="ALL",
        dominant_output_label_match="ANY",
    )

    assert result == (characterizations[0],)
    assert result[0] is characterizations[0]


def test_lateral_filter_excludes_exact_dominant_labels_by_category(monkeypatch):
    characterizations = (
        filtered_characterization(
            True,
            "decaying",
            states=("v", "r"),
            inputs=("delta_a",),
            outputs=("v",),
        ),
        filtered_characterization(
            True,
            "decaying",
            states=("p",),
            inputs=("delta_r",),
            outputs=("p",),
        ),
        filtered_characterization(
            True,
            "decaying",
            states=("phi",),
            inputs=(),
            outputs=(),
        ),
        filtered_characterization(
            False,
            "decaying",
            states=("v",),
            inputs=("delta_r",),
            outputs=("phi",),
        ),
    )
    monkeypatch.setattr(
        LateralDirectionalModel,
        "modal_family_characterizations",
        lambda self: characterizations,
    )
    model = LateralDirectionalModel(**valid_parameters())

    states = model.filter_modal_family_characterizations(
        exclude_dominant_state_labels=("v", "v"),
        exclude_dominant_label_match="EXACT",
        exclude_dominant_state_label_match="ALL",
    )
    inputs = model.filter_modal_family_characterizations(
        exclude_dominant_input_labels="delta_r",
        exclude_dominant_label_match="EXACT",
    )
    outputs = model.filter_modal_family_characterizations(
        exclude_dominant_output_labels="p",
        exclude_dominant_label_match="ALL",
        exclude_dominant_output_label_match="EXACT",
    )

    assert states == characterizations[1:3]
    assert inputs == (characterizations[0], characterizations[2])
    assert outputs == (characterizations[0], characterizations[2], characterizations[3])
    assert all(actual is expected for actual, expected in zip(states, characterizations[1:3], strict=True))


def test_lateral_filter_combines_exclusions_with_existing_filters(monkeypatch):
    characterizations = (
        filtered_characterization(
            True,
            "decaying",
            states=("v", "r"),
            inputs=("delta_a", "delta_r"),
            outputs=("v", "r"),
        ),
        filtered_characterization(
            True,
            "decaying",
            states=("v", "r", "phi"),
            inputs=("delta_a", "delta_r"),
            outputs=("v", "phi"),
        ),
    )
    monkeypatch.setattr(
        LateralDirectionalModel,
        "modal_family_characterizations",
        lambda self: characterizations,
    )

    result = LateralDirectionalModel(
        **valid_parameters()
    ).filter_modal_family_characterizations(
        oscillatory=True,
        stability="decaying",
        dominant_state_labels=("v", "r"),
        dominant_input_labels="delta_a",
        dominant_output_labels="v",
        dominant_label_match="EXACT",
        dominant_state_label_match="ALL",
        dominant_input_label_match="ALL",
        dominant_output_label_match="ANY",
        exclude_dominant_output_labels="r",
        exclude_dominant_label_match="EXACT",
        exclude_dominant_output_label_match="ANY",
    )

    assert result == (characterizations[1],)
    assert result[0] is characterizations[1]


def test_lateral_directional_model_simulates_small_aileron_step():
    model = LateralDirectionalModel(
        trim_speed=10.0,
        trim_pitch=0.1,
        gravity=9.81,
        y_v=-0.9,
        y_p=0.4,
        y_r=7.0,
        y_delta_a=0.2,
        y_delta_r=0.1,
        l_v=-0.8,
        l_p=-2.9,
        l_r=0.3,
        l_delta_a=0.5,
        l_delta_r=-0.2,
        n_v=0.3,
        n_p=0.3,
        n_r=-1.5,
        n_delta_a=-0.1,
        n_delta_r=0.4,
    )
    time = np.linspace(0.0, 0.5, 11)

    states, outputs = model.to_state_space().simulate(
        np.zeros(4), np.array([0.01, 0.0]), time
    )

    assert states.shape == (time.size, 4)
    assert outputs.shape == (time.size, 4)
    assert np.all(np.isfinite(states))
    assert np.all(np.isfinite(outputs))
    assert np.any(np.abs(states[1:]) > 0.0)
    np.testing.assert_allclose(outputs, states)
    phi_rate = states[:-1, 1] + np.tan(model.trim_pitch) * states[:-1, 2]
    np.testing.assert_allclose(np.diff(states[:, 3]), np.diff(time) * phi_rate)

    rk4_states, rk4_outputs = model.to_state_space().simulate(
        np.zeros(4), np.array([0.01, 0.0]), time, method="rk4"
    )
    assert rk4_states.shape == (time.size, 4)
    assert np.all(np.isfinite(rk4_states))
    np.testing.assert_allclose(rk4_outputs, rk4_states)

    zero_input_states, zero_input_outputs = model.to_state_space().zero_input_response(
        np.array([0.1, 0.0, 0.0, 0.01]), time
    )
    assert zero_input_states.shape == (time.size, 4)
    assert np.all(np.isfinite(zero_input_states))
    np.testing.assert_allclose(zero_input_outputs, zero_input_states)

    forced_states, forced_outputs = model.to_state_space().forced_response(
        np.array([0.01, 0.0]), time
    )
    assert forced_states.shape == (time.size, 4)
    assert forced_outputs.shape == (time.size, 4)
    assert np.all(np.isfinite(forced_states))
    assert np.all(np.isfinite(forced_outputs))
    assert np.any(np.abs(forced_states[1:]) > 0.0)
    np.testing.assert_allclose(forced_outputs, forced_states)

    step_states, step_outputs = model.to_state_space().step_response(
        np.array([0.01, 0.0]), time
    )
    assert step_states.shape == (time.size, 4)
    assert step_outputs.shape == (time.size, 4)
    assert np.all(np.isfinite(step_states))
    assert np.all(np.isfinite(step_outputs))
    assert np.any(np.abs(step_states[1:]) > 0.0)
    np.testing.assert_allclose(step_outputs, step_states)

    impulse_states, impulse_outputs = model.to_state_space().impulse_response(
        np.array([0.01, 0.0]), time
    )
    assert impulse_states.shape == (time.size, 4)
    assert impulse_outputs.shape == (time.size, 4)
    assert np.all(np.isfinite(impulse_states))
    assert np.all(np.isfinite(impulse_outputs))
    assert np.any(np.abs(impulse_states[1:]) > 0.0)
    np.testing.assert_allclose(impulse_outputs, impulse_states)


def test_lateral_directional_model_rejects_invalid_parameter():
    parameters = valid_parameters()
    parameters["n_r"] = np.nan

    with np.testing.assert_raises_regex(ValueError, "n_r must be a finite scalar"):
        LateralDirectionalModel(**parameters)
