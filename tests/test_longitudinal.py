from types import SimpleNamespace

import numpy as np
import pytest

from flightlab.longitudinal import LongitudinalModeIdentification, LongitudinalModel
from flightlab.state_space import StateSpace


def valid_parameters():
    return {
        "trim_speed": 10,
        "trim_pitch": 0,
        "gravity": 9,
        "x_u": 1,
        "x_w": 2,
        "x_q": 3,
        "x_delta_e": 4,
        "z_u": 5,
        "z_w": 6,
        "z_q": 7,
        "z_delta_e": 8,
        "m_u": 9,
        "m_w": 10,
        "m_q": 11,
        "m_delta_e": 12,
    }


def generic_characterization(
    dominant_indices,
    values=None,
    input_values=(0.75,),
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


def physical_mode_characterization(
    frequency, period, dominant_states, participation, oscillatory=True
):
    return SimpleNamespace(
        characterization=SimpleNamespace(
            dynamics=SimpleNamespace(
                is_oscillatory=oscillatory,
                natural_frequency=frequency,
                period=period,
            ),
            state_participation=SimpleNamespace(
                participation_magnitudes=np.asarray(participation)
            ),
        ),
        dominant_state_labels=dominant_states,
        state_participation_by_label=tuple(
            zip(LongitudinalModel.STATE_ORDER, participation, strict=True)
        ),
    )


def test_longitudinal_model_builds_expected_state_space_matrices():
    model = LongitudinalModel(**valid_parameters())

    system = model.to_state_space()

    assert isinstance(system, StateSpace)
    np.testing.assert_array_equal(
        system.A,
        [[1, 2, 3, -9], [5, 6, 17, 0], [9, 10, 11, 0], [0, 0, 1, 0]],
    )
    np.testing.assert_array_equal(system.B, [[4], [8], [12], [0]])
    np.testing.assert_array_equal(system.C, np.eye(4))
    np.testing.assert_array_equal(system.D, np.zeros((4, 1)))
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
    assert modal_input.shape == (4, 1)
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
    assert LongitudinalModel.INPUT_ORDER == ("delta_e",)
    assert LongitudinalModel.OUTPUT_ORDER == LongitudinalModel.STATE_ORDER
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
    assert system.rk4_step(np.zeros(4), np.zeros(1), 0.01).shape == (4,)


def test_longitudinal_model_declares_state_input_and_output_ordering():
    assert LongitudinalModel.STATE_ORDER == ("u", "w", "q", "theta")
    assert LongitudinalModel.INPUT_ORDER == ("delta_e",)
    assert LongitudinalModel.OUTPUT_ORDER == LongitudinalModel.STATE_ORDER


def test_longitudinal_identifies_clear_short_period_and_phugoid_modes(monkeypatch):
    phugoid = physical_mode_characterization(
        0.2, 10.0 * np.pi, ("u",), (0.7, 0.05, 0.05, 0.2)
    )
    ambiguous = physical_mode_characterization(
        0.8, 2.5 * np.pi, ("u", "q"), (0.4, 0.1, 0.4, 0.1)
    )
    short_period = physical_mode_characterization(
        2.0, np.pi, ("q",), (0.05, 0.25, 0.65, 0.05)
    )
    characterizations = (phugoid, ambiguous, short_period)
    monkeypatch.setattr(
        LongitudinalModel,
        "modal_family_characterizations",
        lambda self: characterizations,
    )

    result = LongitudinalModel(**valid_parameters()).physical_mode_identifications()

    assert all(isinstance(item, LongitudinalModeIdentification) for item in result)
    assert tuple(item.characterization for item in result) == characterizations
    assert tuple(item.mode_name for item in result) == (
        "phugoid",
        None,
        "short_period",
    )


def test_longitudinal_leaves_state_ambiguous_mode_unclassified(monkeypatch):
    characterizations = (
        physical_mode_characterization(
            0.2, 10.0 * np.pi, ("u",), (0.7, 0.05, 0.05, 0.2)
        ),
        physical_mode_characterization(
            2.0, np.pi, ("u", "q"), (0.4, 0.1, 0.4, 0.1)
        ),
    )
    monkeypatch.setattr(
        LongitudinalModel,
        "modal_family_characterizations",
        lambda self: characterizations,
    )

    result = LongitudinalModel(**valid_parameters()).physical_mode_identifications()

    assert tuple(item.mode_name for item in result) == ("phugoid", None)


@pytest.mark.parametrize(
    ("index", "label"), tuple(enumerate(LongitudinalModel.STATE_ORDER))
)
def test_longitudinal_modal_family_characterization_maps_dominant_state(
    monkeypatch, index, label
):
    generic = generic_characterization((index,))
    monkeypatch.setattr(
        LongitudinalModel,
        "to_state_space",
        lambda self: FakeStateSpace((generic,)),
    )

    (interpreted,) = LongitudinalModel(**valid_parameters()).modal_family_characterizations()

    assert interpreted.characterization is generic
    assert interpreted.state_labels == ("u", "w", "q", "theta")
    assert interpreted.dominant_state_labels == (label,)
    assert interpreted.characterization.dynamics is generic.dynamics
    assert interpreted.characterization.input_influence is generic.input_influence
    assert interpreted.characterization.output_influence is generic.output_influence


def test_longitudinal_modal_family_characterization_maps_input_and_outputs(
    monkeypatch,
):
    generic = generic_characterization(
        (0,), output_values=(0.8, 0.2, 0.8, 0.1), dominant_output_indices=(0, 2)
    )
    monkeypatch.setattr(
        LongitudinalModel,
        "to_state_space",
        lambda self: FakeStateSpace((generic,)),
    )

    (interpreted,) = LongitudinalModel(
        **valid_parameters()
    ).modal_family_characterizations()

    assert interpreted.input_labels == LongitudinalModel.INPUT_ORDER
    assert interpreted.dominant_input_labels == ("delta_e",)
    assert interpreted.input_influence_by_label == (("delta_e", 0.75),)
    assert interpreted.output_labels == LongitudinalModel.OUTPUT_ORDER
    assert interpreted.dominant_output_labels == ("u", "q")
    assert interpreted.output_influence_by_label == (
        ("u", 0.8),
        ("w", 0.2),
        ("q", 0.8),
        ("theta", 0.1),
    )


def test_longitudinal_modal_family_characterization_preserves_zero_channels(
    monkeypatch,
):
    generic = generic_characterization(
        (0,), dominant_input_indices=(), dominant_output_indices=()
    )
    monkeypatch.setattr(
        LongitudinalModel,
        "to_state_space",
        lambda self: FakeStateSpace((generic,)),
    )

    (interpreted,) = LongitudinalModel(
        **valid_parameters()
    ).modal_family_characterizations()

    assert interpreted.dominant_input_labels == ()
    assert interpreted.dominant_output_labels == ()


def test_longitudinal_modal_family_characterization_preserves_tie_and_order(
    monkeypatch,
):
    first = generic_characterization((0, 2, 3), [0.4, 0.1, 0.4, 0.4])
    second = generic_characterization((1,), [0.1, 0.7, 0.1, 0.1])
    monkeypatch.setattr(
        LongitudinalModel,
        "to_state_space",
        lambda self: FakeStateSpace((first, second)),
    )

    interpreted = LongitudinalModel(**valid_parameters()).modal_family_characterizations()

    assert [result.characterization for result in interpreted] == [first, second]
    assert interpreted[0].dominant_state_labels == ("u", "q", "theta")
    assert interpreted[1].dominant_state_labels == ("w",)
    assert interpreted[0].state_participation_by_label == (
        ("u", 0.4),
        ("w", 0.1),
        ("q", 0.4),
        ("theta", 0.4),
    )


@pytest.mark.parametrize(
    ("indices", "values", "message"),
    [
        ((4,), np.full(4, 0.25), "dominant state index is invalid"),
        ((0,), np.full(3, 1 / 3), "must match the model state dimension"),
    ],
)
def test_longitudinal_modal_family_characterization_validates_state_mapping(
    monkeypatch, indices, values, message
):
    generic = generic_characterization(indices, values)
    monkeypatch.setattr(
        LongitudinalModel,
        "to_state_space",
        lambda self: FakeStateSpace((generic,)),
    )

    with pytest.raises(ValueError, match=message):
        LongitudinalModel(**valid_parameters()).modal_family_characterizations()


@pytest.mark.parametrize(
    ("filters", "expected_indices"),
    [
        ({"oscillatory": True}, (1, 3)),
        ({"oscillatory": False}, (0, 2)),
        ({"oscillatory": None}, (0, 1, 2, 3)),
        ({"stability": "decaying"}, (0, 1)),
        ({"stability": "growing"}, (2,)),
        ({"stability": "neutral"}, (3,)),
        ({"stability": None}, (0, 1, 2, 3)),
        ({"oscillatory": True, "stability": "decaying"}, (1,)),
        ({"oscillatory": False, "stability": "neutral"}, ()),
    ],
)
def test_longitudinal_filters_modal_family_characterizations(
    monkeypatch, filters, expected_indices
):
    characterizations = (
        filtered_characterization(False, "decaying"),
        filtered_characterization(True, "decaying"),
        filtered_characterization(False, "growing"),
        filtered_characterization(True, "neutral"),
    )
    monkeypatch.setattr(
        LongitudinalModel,
        "modal_family_characterizations",
        lambda self: characterizations,
    )

    result = LongitudinalModel(
        **valid_parameters()
    ).filter_modal_family_characterizations(**filters)

    assert result == tuple(characterizations[index] for index in expected_indices)
    assert all(
        actual is characterizations[index]
        for actual, index in zip(result, expected_indices, strict=True)
    )


@pytest.mark.parametrize(
    ("filters", "message"),
    [
        ({"oscillatory": "yes"}, "oscillatory must be True, False, or None"),
        ({"stability": "stable"}, "stability must be"),
    ],
)
def test_longitudinal_filter_rejects_invalid_values(monkeypatch, filters, message):
    monkeypatch.setattr(
        LongitudinalModel,
        "modal_family_characterizations",
        lambda self: (),
    )

    with pytest.raises(ValueError, match=message):
        LongitudinalModel(
            **valid_parameters()
        ).filter_modal_family_characterizations(**filters)


@pytest.mark.parametrize("label", LongitudinalModel.STATE_ORDER)
def test_longitudinal_filter_matches_each_dominant_state_label(monkeypatch, label):
    characterizations = tuple(
        filtered_characterization(False, "decaying", states=(state_label,))
        for state_label in LongitudinalModel.STATE_ORDER
    )
    monkeypatch.setattr(
        LongitudinalModel,
        "modal_family_characterizations",
        lambda self: characterizations,
    )

    result = LongitudinalModel(
        **valid_parameters()
    ).filter_modal_family_characterizations(dominant_state_labels=label)

    expected = characterizations[LongitudinalModel.STATE_ORDER.index(label)]
    assert result == (expected,)
    assert result[0] is expected


def test_longitudinal_filter_uses_any_labels_and_and_categories(monkeypatch):
    characterizations = (
        filtered_characterization(
            True,
            "decaying",
            states=("u",),
            inputs=("delta_e",),
            outputs=("theta",),
        ),
        filtered_characterization(
            True,
            "decaying",
            states=("w", "q"),
            inputs=("delta_e",),
            outputs=("q",),
        ),
        filtered_characterization(
            False,
            "growing",
            states=("theta",),
            inputs=(),
            outputs=(),
        ),
    )
    monkeypatch.setattr(
        LongitudinalModel,
        "modal_family_characterizations",
        lambda self: characterizations,
    )
    model = LongitudinalModel(**valid_parameters())

    any_states = model.filter_modal_family_characterizations(
        dominant_state_labels=("u", "q", "q")
    )
    combined = model.filter_modal_family_characterizations(
        oscillatory=True,
        stability="decaying",
        dominant_state_labels=("w", "q"),
        dominant_input_labels="delta_e",
        dominant_output_labels="q",
    )
    zero_dominance = model.filter_modal_family_characterizations(
        dominant_input_labels="delta_e", stability="growing"
    )

    assert any_states == characterizations[:2]
    assert combined == (characterizations[1],)
    assert combined[0] is characterizations[1]
    assert zero_dominance == ()


@pytest.mark.parametrize(
    ("match", "expected_indices"),
    [
        ("ANY", (0, 1, 2)),
        ("ALL", (1, 2)),
        ("EXACT", (1,)),
    ],
)
def test_longitudinal_filter_configures_dominant_label_set_matching(
    monkeypatch, match, expected_indices
):
    characterizations = (
        filtered_characterization(False, "decaying", states=("u",)),
        filtered_characterization(False, "decaying", states=("u", "q")),
        filtered_characterization(False, "decaying", states=("u", "q", "theta")),
        filtered_characterization(False, "decaying", states=("w",)),
    )
    monkeypatch.setattr(
        LongitudinalModel,
        "modal_family_characterizations",
        lambda self: characterizations,
    )

    result = LongitudinalModel(
        **valid_parameters()
    ).filter_modal_family_characterizations(
        dominant_state_labels=("q", "u", "u"),
        dominant_label_match=match,
    )

    assert result == tuple(characterizations[index] for index in expected_indices)
    assert all(
        actual is characterizations[index]
        for actual, index in zip(result, expected_indices, strict=True)
    )


@pytest.mark.parametrize(
    ("match", "expected_indices"),
    [
        ("ANY", (3,)),
        ("ALL", (0, 3)),
        ("EXACT", (0, 2, 3)),
    ],
)
def test_longitudinal_filter_configures_dominant_label_exclusion_set_matching(
    monkeypatch, match, expected_indices
):
    characterizations = (
        filtered_characterization(False, "decaying", states=("u",)),
        filtered_characterization(False, "decaying", states=("u", "q")),
        filtered_characterization(False, "decaying", states=("u", "q", "theta")),
        filtered_characterization(False, "decaying", states=("w",)),
    )
    monkeypatch.setattr(
        LongitudinalModel,
        "modal_family_characterizations",
        lambda self: characterizations,
    )

    result = LongitudinalModel(
        **valid_parameters()
    ).filter_modal_family_characterizations(
        exclude_dominant_state_labels=("q", "u", "u"),
        exclude_dominant_label_match=match,
    )

    assert result == tuple(characterizations[index] for index in expected_indices)
    assert all(
        actual is characterizations[index]
        for actual, index in zip(result, expected_indices, strict=True)
    )


@pytest.mark.parametrize(
    ("filters", "message"),
    [
        ({"dominant_state_labels": "airspeed"}, "invalid dominant state label"),
        ({"dominant_input_labels": "elevator"}, "invalid dominant input label"),
        ({"dominant_output_labels": "pitch rate"}, "invalid dominant output label"),
        ({"dominant_state_labels": ()}, "state label filter must not be empty"),
        ({"dominant_label_match": "all"}, "dominant_label_match must be"),
        (
            {"dominant_state_label_match": "all"},
            "dominant_state_label_match must be",
        ),
        (
            {"dominant_input_label_match": "all"},
            "dominant_input_label_match must be",
        ),
        (
            {"dominant_output_label_match": "all"},
            "dominant_output_label_match must be",
        ),
        (
            {"exclude_dominant_state_labels": "airspeed"},
            "invalid excluded dominant state label",
        ),
        (
            {"exclude_dominant_input_labels": "elevator"},
            "invalid excluded dominant input label",
        ),
        (
            {"exclude_dominant_output_labels": "pitch rate"},
            "invalid excluded dominant output label",
        ),
        (
            {"exclude_dominant_state_labels": ()},
            "excluded dominant state label filter must not be empty",
        ),
        (
            {"exclude_dominant_label_match": "all"},
            "exclude_dominant_label_match must be",
        ),
        (
            {"exclude_dominant_state_label_match": "all"},
            "exclude_dominant_state_label_match must be",
        ),
        (
            {"exclude_dominant_input_label_match": "all"},
            "exclude_dominant_input_label_match must be",
        ),
        (
            {"exclude_dominant_output_label_match": "all"},
            "exclude_dominant_output_label_match must be",
        ),
    ],
)
def test_longitudinal_filter_rejects_invalid_label_filters(
    monkeypatch, filters, message
):
    monkeypatch.setattr(
        LongitudinalModel,
        "modal_family_characterizations",
        lambda self: (),
    )

    with pytest.raises(ValueError, match=message):
        LongitudinalModel(
            **valid_parameters()
        ).filter_modal_family_characterizations(**filters)


def test_longitudinal_model_simulates_small_elevator_step():
    model = LongitudinalModel(
        trim_speed=10.0,
        trim_pitch=0.0,
        gravity=9.81,
        x_u=-0.4,
        x_w=-0.2,
        x_q=0.1,
        x_delta_e=0.2,
        z_u=-0.3,
        z_w=-1.8,
        z_q=-1.7,
        z_delta_e=0.5,
        m_u=0.9,
        m_w=-0.9,
        m_q=-2.7,
        m_delta_e=-1.0,
    )
    time = np.linspace(0.0, 0.5, 11)

    states, outputs = model.to_state_space().simulate(
        np.zeros(4), np.array([0.01]), time
    )

    assert states.shape == (time.size, 4)
    assert outputs.shape == (time.size, 4)
    assert np.all(np.isfinite(states))
    assert np.all(np.isfinite(outputs))
    assert np.any(np.abs(states[1:]) > 0.0)
    np.testing.assert_allclose(outputs, states)
    np.testing.assert_allclose(
        np.diff(states[:, 3]), np.diff(time) * states[:-1, 2]
    )

    rk4_states, rk4_outputs = model.to_state_space().simulate(
        np.zeros(4), np.array([0.01]), time, method="rk4"
    )
    assert rk4_states.shape == (time.size, 4)
    assert np.all(np.isfinite(rk4_states))
    np.testing.assert_allclose(rk4_outputs, rk4_states)

    zero_input_states, zero_input_outputs = model.to_state_space().zero_input_response(
        np.array([0.1, 0.0, 0.0, 0.01]), time, method="rk4"
    )
    assert zero_input_states.shape == (time.size, 4)
    assert np.all(np.isfinite(zero_input_states))
    np.testing.assert_allclose(zero_input_outputs, zero_input_states)

    forced_states, forced_outputs = model.to_state_space().forced_response(
        np.array([0.01]), time
    )
    assert forced_states.shape == (time.size, 4)
    assert forced_outputs.shape == (time.size, 4)
    assert np.all(np.isfinite(forced_states))
    assert np.all(np.isfinite(forced_outputs))
    assert np.any(np.abs(forced_states[1:]) > 0.0)
    np.testing.assert_allclose(forced_outputs, forced_states)

    step_states, step_outputs = model.to_state_space().step_response(
        np.array([0.01]), time
    )
    assert step_states.shape == (time.size, 4)
    assert step_outputs.shape == (time.size, 4)
    assert np.all(np.isfinite(step_states))
    assert np.all(np.isfinite(step_outputs))
    assert np.any(np.abs(step_states[1:]) > 0.0)
    np.testing.assert_allclose(step_outputs, step_states)

    impulse_states, impulse_outputs = model.to_state_space().impulse_response(
        np.array([0.01]), time
    )
    assert impulse_states.shape == (time.size, 4)
    assert impulse_outputs.shape == (time.size, 4)
    assert np.all(np.isfinite(impulse_states))
    assert np.all(np.isfinite(impulse_outputs))
    assert np.any(np.abs(impulse_states[1:]) > 0.0)
    np.testing.assert_allclose(impulse_outputs, impulse_states)


def test_longitudinal_model_rejects_non_finite_parameter():
    parameters = valid_parameters()
    parameters["m_q"] = np.nan

    with np.testing.assert_raises_regex(ValueError, "m_q must be a finite scalar"):
        LongitudinalModel(**parameters)


def test_longitudinal_model_rejects_infinite_parameter():
    parameters = valid_parameters()
    parameters["trim_speed"] = np.inf

    with np.testing.assert_raises_regex(
        ValueError, "trim_speed must be a finite scalar"
    ):
        LongitudinalModel(**parameters)


def test_longitudinal_model_rejects_array_parameter():
    parameters = valid_parameters()
    parameters["x_u"] = np.array(1.0)

    with np.testing.assert_raises_regex(ValueError, "x_u must be a finite scalar"):
        LongitudinalModel(**parameters)


def test_longitudinal_model_rejects_non_scalar_parameter():
    parameters = valid_parameters()
    parameters["z_w"] = [1.0]

    with np.testing.assert_raises_regex(ValueError, "z_w must be a finite scalar"):
        LongitudinalModel(**parameters)
