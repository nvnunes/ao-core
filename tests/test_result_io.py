"""Check shared output storage independently of simulator code."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from astropy import units as u

from ao_core.io import (
    clear_observation,
    collect_schema_issues,
    initialize_result,
    read_result,
    write_observation,
)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_result_io_stores_outputs_and_declarations(tmp_path, batched, precision):
    path = tmp_path / "result.h5"
    wavelength = (np.array([1.25]) if batched else 1.25) * u.um
    index = 0 if batched else None
    simulation = {
        "precision": precision,
        "extra_stat_fields": {"jitter": "mas"},
        "meta_fields": {"exposure": "s", "count": None},
        "diagnostic_fields": {
            "quality/covariance": {
                "dtype": "float64",
                "unit": "mas2",
                "shape": ("num_sci", 2, 2),
            },
            "quality/label": {"dtype": "str", "shape": ()},
            "quality/point_name": {"dtype": "str", "shape": ("num_sci",)},
        },
    }
    setup = {
        "sci_r": np.array([0.0, 10.0]) * u.arcsec,
        "sci_theta": np.array([0.0, 90.0]) * u.deg,
    }
    diagnostics = {
        "quality": {
            "covariance": np.broadcast_to(np.eye(2), (2, 2, 2)) * u.mas**2,
            "label": "résultat",
            "point_name": ["étoile", "α"],
        }
    }
    with h5py.File(path, "w") as file:
        initialize_result(
            file,
            simulation=simulation,
            setup=setup,
            options={"wavelength": wavelength},
            save_psfs=True,
        )
        assert read_result(file, index=index, sections=("psfs", "meta"))["psfs"] == {}
        assert file["meta/tel_pupil"].shape == (0, 0)
        write_observation(
            file,
            index=index,
            psfs=np.ones((2, 3, 5)),
            meta={
                "pixel_scale": 4.0 * u.mas,
                "tel_diameter": 8.0 * u.m,
                "tel_pupil": np.ones((6, 6)) * u.one,
                "exposure": 2.0 * u.s,
                "count": 9,
            },
            stats={"jitter": np.array([1.0, 2.0]) * u.mas},
            diagnostics=diagnostics,
        )
        assert collect_schema_issues(file) == []
        assert file["psfs/data"].shape == ((1,) if batched else ()) + (2, 3, 5)
        assert file["psfs/data"].dtype == np.dtype(precision)
        assert file["simulation/diagnostic_fields/quality/covariance/shape"].asstr()[
            ...
        ].tolist() == [
            "num_sci",
            "2",
            "2",
        ]
        assert "units" not in file["meta/count"].attrs
    with h5py.File(path, "r") as file:
        row = read_result(file, index=index)
        np.testing.assert_array_equal(
            row["stats"]["jitter"].to_value(u.mas), [1.0, 2.0]
        )
        assert row["meta"]["exposure"] == 2.0 * u.s
        assert row["meta"]["count"] == 9
        assert row["diagnostics"]["quality"]["label"] == "résultat"
        assert list(row["diagnostics"]["quality"]["point_name"]) == ["étoile", "α"]
        assert row["options"]["wavelength"].isscalar
        assert row["psfs"]["data"].shape == (2, 3, 5)
        assert row["meta"]["tel_pupil"].shape == (6, 6)


def test_selected_read_and_psf_free_metadata(tmp_path):
    with h5py.File(tmp_path / "batch.h5", "w") as file:
        initialize_result(
            file, simulation={}, setup={}, options={"wavelength": [1.25] * u.um}
        )
        with pytest.raises(ValueError, match="requires an observation index"):
            write_observation(file, meta={})
        write_observation(file, meta={"pixel_scale": 4.0 * u.mas}, index=0)
        assert read_result(file, index=0)["meta"]["pixel_scale"] == 4.0 * u.mas
        clear_observation(file, index=0)
        assert np.isnan(read_result(file, index=0)["meta"]["pixel_scale"].value)
        assert "psfs" not in file
        del file["meta/pixel_scale"].attrs["units"]
        assert (
            read_result(file, index=0, sections=("options",))["options"]["wavelength"]
            == 1.25 * u.um
        )
        assert collect_schema_issues(file) == [
            "/meta/pixel_scale must declare units='mas'."
        ]
        with pytest.raises(ValueError, match="pixel_scale"):
            read_result(file, index=0, sections=("meta",))


def test_rejected_psf_shape_preserves_written_row(tmp_path):
    with h5py.File(tmp_path / "batch.h5", "w") as file:
        initialize_result(
            file,
            simulation={"extra_stat_fields": {"jitter": "mas"}},
            setup={"sci_r": [0.0] * u.arcsec, "sci_theta": [0.0] * u.deg},
            options={"wavelength": [1.25] * u.um},
            save_psfs=True,
        )
        meta = {
            "pixel_scale": 4.0 * u.mas,
            "tel_diameter": 8.0 * u.m,
            "tel_pupil": np.ones((6, 6)) * u.one,
        }
        write_observation(
            file,
            index=0,
            meta=meta,
            psfs=np.ones((1, 3, 5)),
            stats={"jitter": [1.0] * u.mas},
        )
        with pytest.raises(ValueError, match="/psfs/data shape mismatch"):
            write_observation(
                file,
                index=0,
                meta={**meta, "pixel_scale": 5.0 * u.mas},
                psfs=np.ones((1, 4, 4)),
                stats={"jitter": [2.0] * u.mas},
            )
        row = read_result(file, index=0)
        assert row["meta"]["pixel_scale"] == 4.0 * u.mas
        assert row["stats"]["jitter"][0] == 1.0 * u.mas

        del file["meta"]
        with pytest.raises(ValueError, match="/meta/pixel_scale"):
            read_result(file, index=0, sections=("psfs",))


@pytest.mark.parametrize(
    "options",
    [{"producer": {"a": 4}}, {"/options/wavelength": "overwrite"}],
)
def test_initialize_rejects_options_that_are_not_direct_datasets(tmp_path, options):
    with h5py.File(tmp_path / "result.h5", "w") as file:
        with pytest.raises(ValueError, match="must be a direct dataset"):
            initialize_result(
                file,
                simulation={},
                setup={},
                options={"wavelength": 1.25 * u.um, **options},
            )
        assert len(file) == 0


def test_schema_rejects_text_wavelength(tmp_path):
    with h5py.File(tmp_path / "result.h5", "w") as file:
        file.create_dataset("options/wavelength", data="1.25").attrs["units"] = "um"
        assert collect_schema_issues(file) == [
            "/options/wavelength must contain real numbers."
        ]


@pytest.mark.parametrize(
    "options, diagnostic_fields, message",
    [
        ({"wavelength": (1.25 + 1j) * u.um}, {}, "wavelength must contain real"),
        ({}, {"group/./value": {"dtype": "bool"}}, "Invalid diagnostic field path"),
    ],
)
def test_initialize_rejects_invalid_inputs_before_writing(
    tmp_path, options, diagnostic_fields, message
):
    with h5py.File(tmp_path / "result.h5", "w") as file:
        with pytest.raises(ValueError, match=message):
            initialize_result(
                file,
                simulation={"diagnostic_fields": diagnostic_fields},
                setup={},
                options={"wavelength": 1.25 * u.um, **options},
            )
        assert not file.keys()


def test_selected_read_rejects_a_wavelength_without_a_dataspace(tmp_path):
    with h5py.File(tmp_path / "result.h5", "w") as file:
        file.create_dataset("options/wavelength", dtype=float).attrs["units"] = "um"
        with pytest.raises(ValueError, match="wavelength must be scalar"):
            read_result(file, sections=("stats",))


def test_schema_checks_required_types_of_physical_inputs(tmp_path):
    with h5py.File(tmp_path / "result.h5", "w") as file:
        file.create_dataset("options/wavelength", data=1.25).attrs["units"] = "um"
        file.create_dataset("options/zenith_angle", data="20").attrs["units"] = "deg"
        file.create_dataset("setup/sci_r/group_contents", data=0.0)
        file.create_dataset("setup/sci_theta", data=[0.0]).attrs["units"] = "deg"
        issues = collect_schema_issues(file)
        assert "/options/zenith_angle must contain real numbers." in issues
        assert "/setup/sci_r must be a dataset." in issues


def test_selected_read_checks_only_requested_outputs(tmp_path):
    setup = {"sci_r": [0.0] * u.arcsec, "sci_theta": [0.0] * u.deg}
    with h5py.File(tmp_path / "result.h5", "w") as file:
        initialize_result(
            file,
            simulation={"precision": "float32"},
            setup=setup,
            options={"wavelength": 1.25 * u.um},
            standard_stats=("sr",),
        )
        del file["stats/sr"]
        file.create_dataset("stats/sr", data=[0], dtype="int32").attrs["units"] = "1"
        assert set(read_result(file, sections=("meta",))) == {"meta"}
        assert collect_schema_issues(file) == ["/stats/sr must have dtype float32."]
