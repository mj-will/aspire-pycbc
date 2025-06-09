from pycbc.inference.io.base_sampler import BaseSamplerFile
from pycbc.inference.io.posterior import (
    read_raw_samples_from_file,
    write_samples_to_file,
)


class PoppyFile(BaseSamplerFile):
    """Class to handle file IO for the ``poppy`` sampler."""

    name = "poppy_file"

    def read_raw_samples(self, fields, **kwargs):
        return read_raw_samples_from_file(self, fields, **kwargs)

    def write_samples(self, samples, **kwargs):
        """Write samples to the file."""
        return write_samples_to_file(
            self,
            samples,
            **kwargs,
        )

    def write_sampler_metadata(self, sampler):
        """Write sampler metadata to the file."""
        sampler.model.write_metadata(self)

    def write_resume_point(self):
        pass
