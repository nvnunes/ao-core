"""Check the shared INI behavior used by downstream loaders."""

import ast

import numpy as np
import pytest

from ao_core.models.config import (
    ini_to_mapping,
    parse_ini,
    polar_to_cartesian,
    read_ini,
    read_ini_array,
    read_ini_scalar,
    resolve_ini_path,
)


def test_read_ini_preserves_keys_and_values_and_resolves_from_source(
    tmp_path, monkeypatch
):
    config_dir = tmp_path / "instrument"
    config_dir.mkdir()
    path = config_dir / "system.ini"
    contents = """[telescope]
TelescopeDiameter = 8.0
PathPupil = pupil.fits
Label = 50% transmission
"""
    path.write_text(contents, encoding="utf-8")
    pupil = config_dir / "pupil.fits"
    pupil.touch()
    monkeypatch.chdir(tmp_path)

    source, ini = read_ini("instrument/system.ini")

    assert source == path.resolve()
    telescope = ini["telescope"]
    assert dict(telescope) == dict(parse_ini(contents)["telescope"])
    assert "telescopediameter" not in telescope
    assert telescope["Label"] == "50% transmission"
    pupil_path = resolve_ini_path(
        telescope["PathPupil"], source=source, label="telescope.PathPupil"
    )
    assert pupil_path == pupil.resolve()


def test_path_resolution_checks_required_files_only(tmp_path):
    source = tmp_path / "system.ini"
    with pytest.raises(ValueError, match=r"output must not be empty"):
        resolve_ini_path(" ", source=source, label="output")
    with pytest.raises(FileNotFoundError, match=r"output"):
        resolve_ini_path("results/run.h5", source=source, label="output")
    result = resolve_ini_path(
        "results/run.h5", source=source, label="output", must_exist=False
    )
    assert result == (tmp_path / "results/run.h5").resolve()


def test_numeric_literals_and_coordinate_convention():
    directions = parse_ini(
        "[sources_HO]\nZenith = [2, 3]\nAzimuth = [90, 180]\nHeight = 90e3"
    )["sources_HO"]
    radius = read_ini_array(directions, "Zenith", label="sources_HO.Zenith")
    azimuth = read_ini_array(directions, "Azimuth", label="sources_HO.Azimuth")
    x, y = polar_to_cartesian(radius, azimuth)
    np.testing.assert_allclose(x, [0, -3], rtol=0, atol=1e-14)
    np.testing.assert_allclose(y, [2, 0], rtol=0, atol=1e-14)
    assert radius.dtype == np.float64
    assert read_ini_scalar(directions, "Height", label="sources_HO.Height") == 90_000
    np.testing.assert_array_equal(
        read_ini_array(directions, "Height", label="sources_HO.Height"),
        [90_000.0],
    )
    with pytest.raises(ValueError, match="identical shapes"):
        polar_to_cartesian(radius, azimuth[:1])


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ({}, "requires atmosphere.Cn2Weights"),
        ({"Cn2Weights": "2*0.5"}, "must be a numeric scalar or list"),
        ({"Cn2Weights": "[]"}, "non-empty one-dimensional list"),
        ({"Cn2Weights": "[[1.0]]"}, "non-empty one-dimensional list"),
        ({"Cn2Weights": "[1e309]"}, "must contain only finite values"),
    ],
)
def test_array_errors_identify_the_field(section, message):
    with pytest.raises(ValueError, match=message):
        read_ini_array(section, "Cn2Weights", label="atmosphere.Cn2Weights")


@pytest.mark.parametrize("section", [{}, {"Height": "[90000]"}, {"Height": "NaN"}])
def test_scalar_errors_identify_the_field(section):
    with pytest.raises(ValueError, match=r"sources_HO\.Height"):
        read_ini_scalar(section, "Height", label="sources_HO.Height")


def test_handoff_selects_sections_and_owns_decoded_values():
    ini = parse_ini(
        "[DEFAULT]\nWavelength = 5e-7\n"
        "[atmosphere]\nCn2Heights = [0, 1000]\n"
        "[application]\nFile = some/file"
    )
    heights = [0, 1000]

    def decode(section, option, raw_value):
        assert section == "atmosphere"
        if option == "Cn2Heights":
            return heights
        return ast.literal_eval(raw_value)

    first = ini_to_mapping(ini, decode, sections=("atmosphere",))
    second = ini_to_mapping(ini, decode, sections=("atmosphere",))
    assert first == {"atmosphere": {"Wavelength": 5e-7, "Cn2Heights": heights}}
    first["atmosphere"]["Cn2Heights"][0] = 1e-6
    assert heights == [0, 1000]
    assert second["atmosphere"]["Cn2Heights"] == [0, 1000]
    assert ini["atmosphere"]["Cn2Heights"] == "[0, 1000]"


def test_handoff_copies_each_decoded_value_before_reusing_scratch_space():
    ini = parse_ini("[telescope]\nFirst = 1\nSecond = 2")
    scratch = np.empty(1)

    def decode(_section, _option, raw_value):
        scratch[0] = float(raw_value)
        return scratch

    values = ini_to_mapping(ini, decode)["telescope"]
    np.testing.assert_array_equal(values["First"], [1.0])
    np.testing.assert_array_equal(values["Second"], [2.0])
    values["Second"][0] = 3.0
    np.testing.assert_array_equal(scratch, [2.0])
