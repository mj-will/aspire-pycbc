from typing import Any, Dict, Optional

from poppy import Poppy
from poppy.samples import Samples
from pycbc.inference.sampler.base import BaseSampler, setup_output
from pycbc.pool import choose_pool

from .io import PoppyFile
from .utils import (
    get_inputs_from_pycbc_model,
    samples_from_pycbc_model,
    samples_from_pycbc_result,
    try_literal_eval,
)


class PoppySampler(BaseSampler):
    """
    PyCBC sampler interface for the Poppy sampler.

    This class wraps the Poppy sampler to be compatible with the PyCBC inference framework.
    """

    name: str = "poppy"
    _io = PoppyFile

    def __init__(
        self,
        model: Any,
        loglikelihood_function: str,
        nprocesses: int = 1,
        use_mpi: bool = False,
        sample_kwds: Optional[Dict[str, Any]] = None,
        fit_kwds: Optional[Dict[str, Any]] = None,
        extra_kwds: Optional[Dict[str, Any]] = None,
        initial_result_file: Optional[str] = None,
        n_samples: int = 1000,
        n_initial_samples: Optional[int] = None,
    ):
        """
        Initialize the PoppySampler.

        Parameters
        ----------
        model : pycbc.inference.model.Model
            The PyCBC model to sample from.
        loglikelihood_function : str
            Name of the loglikelihood function to use.
        nprocesses : int, optional
            Number of processes to use (default: 1).
        use_mpi : bool, optional
            Whether to use MPI for parallelization (default: False).
        sample_kwds : dict, optional
            Additional keyword arguments for sampling.
        fit_kwds : dict, optional
            Additional keyword arguments for fitting.
        extra_kwds : dict, optional
            Additional keyword arguments for the Poppy sampler.
        initial_result_file : str, optional
            Path to a file with initial samples.
        n_samples : int, optional
            Number of samples to draw (default: 1000).
        n_initial_samples : int, optional
            Number of initial samples to draw from the prior.
        """
        super().__init__(model)

        self.sample_kwds = sample_kwds or {}
        self.extra_kwds = extra_kwds or {}
        self.fit_kwds = fit_kwds or {}
        self.initial_result_file = initial_result_file
        self.n_samples = n_samples
        self.n_initial_samples = n_initial_samples

        self.pool = choose_pool(mpi=use_mpi, processes=nprocesses)

        self.inputs = get_inputs_from_pycbc_model(
            model, loglikelihood_function
        )
        self._sampler: Optional[Poppy] = None
        self._samples: Optional[Any] = None

    @property
    def io(self) -> Any:
        """Return the IO class for this sampler."""
        return self._io

    @property
    def model_stats(self) -> None:
        """Return model statistics (not implemented)."""
        pass

    @property
    def samples(self) -> Dict[str, Any]:
        """
        Return the samples from the sampler.

        Returns
        -------
        dict
            Dictionary of samples with keys for parameters and log values.
        Raises
        ------
        RuntimeError
            If the sampler has not been run yet.
        """
        if self._sampler is None or self._samples is None:
            raise RuntimeError("Sampler has not been run yet.")
        samples_dict = self._samples.to_dict()
        samples = {}
        for key in self.model.sampling_params:
            samples[key] = samples_dict[key]
        samples["loglikelihood"] = samples_dict.get("log_likelihood")
        samples["logprior"] = samples_dict.get("log_prior")
        return samples

    def get_initial_samples(self) -> Samples:
        """
        Get initial samples for the sampler.

        Returns
        -------
        Samples
            Initial samples for the Poppy sampler.
        """
        if self.initial_result_file is not None:
            return samples_from_pycbc_result(
                self.initial_result_file, self.inputs.parameters
            )
        else:
            return samples_from_pycbc_model(
                self.model,
                self.n_initial_samples or self.n_samples,
            )

    def run(self) -> None:
        """
        Run the Poppy sampler to generate posterior samples.

        This method initializes the sampler if needed, fits it to the initial samples,
        and then draws posterior samples.
        """
        extra_kwds = self.extra_kwds.copy()

        if self._sampler is None:
            self._sampler = Poppy(
                log_likelihood=self.inputs.log_likelihood,
                log_prior=self.inputs.log_prior,
                dims=self.inputs.dims,
                parameters=self.inputs.parameters,
                prior_bounds=self.inputs.prior_bounds,
                periodic_parameters=self.inputs.periodic_parameters,
                **extra_kwds,
            )

        initial_samples = self.get_initial_samples()

        self._sampler.fit(initial_samples, **self.fit_kwds)

        with self._sampler.enable_pool(self.pool, close_pool=False):
            self._samples = self._sampler.sample_posterior(
                self.n_samples, **self.sample_kwds
            )

    @classmethod
    def from_config(
        cls,
        cp: Any,
        model: Any,
        output_file: Optional[str] = None,
        nprocesses: int = 1,
        use_mpi: bool = False,
    ) -> "PoppySampler":
        """
        Create a PoppySampler from a configuration parser.

        Parameters
        ----------
        cp : configparser.ConfigParser
            Configuration parser.
        model : pycbc.inference.model.Model
            The PyCBC model.
        output_file : str, optional
            Output file path.
        nprocesses : int, optional
            Number of processes to use.
        use_mpi : bool, optional
            Whether to use MPI.

        Returns
        -------
        PoppySampler
            Configured PoppySampler instance.
        """
        section = "sampler"
        if not cp.get(section, "name") == cls.name:
            raise ValueError(
                f"Configuration section '{section}' does not match the expected sampler '{cls.name}'."
            )

        initial_result_file = cp.get(
            section, "initial_result_file", fallback=None
        )
        n_samples = cp.getint(section, "n_samples", fallback=1000)
        n_initial_samples = cp.getint(
            section, "n_initial_samples", fallback=None
        )

        loglikelihood_function = cp.get(
            section,
            "loglikelihood-function",
            fallback="loglikelihood",
        )

        # Extra keyword arguments from the sampler section
        extra_kwds = dict(cp.items(section)) if cp.has_section(section) else {}
        # Remove known parameters from extra_kwds
        known_params = [
            "name",
            "initial_result_file",
            "n_samples",
            "output_file",
            "nprocesses",
            "use_mpi",
            "loglikelihood_function",
        ]
        for param in known_params:
            extra_kwds.pop(param, None)

        section = "sampler.fit"
        fit_kwds = dict(cp.items(section)) if cp.has_section(section) else {}

        section = "sampler.sample"
        sample_kwds = (
            dict(cp.items(section)) if cp.has_section(section) else {}
        )

        # Convert string representations of lists or dicts to actual types
        for kwds in (extra_kwds, fit_kwds, sample_kwds):
            for key, value in kwds.items():
                kwds[key] = try_literal_eval(value)

        obj = cls(
            model=model,
            loglikelihood_function=loglikelihood_function,
            nprocesses=nprocesses,
            use_mpi=use_mpi,
            sample_kwds=sample_kwds,
            extra_kwds=extra_kwds,
            fit_kwds=fit_kwds,
            initial_result_file=initial_result_file,
            n_samples=n_samples,
            n_initial_samples=n_initial_samples,
        )
        setup_output(obj, output_file=output_file)

        return obj

    def resume_from_checkpoint(self):
        raise NotImplementedError(
            "PoppySampler does not support resuming from checkpoints."
        )

    def checkpoint(self):
        raise NotImplementedError(
            "PoppySampler does not support checkpointing."
        )

    def finalize(self):
        """Finalize the sampler and write results to the checkpoint file."""
        self.write_results(self.checkpoint_file)

    def write_results(self, filename: str) -> None:
        """
        Write the results of the sampler to a file.

        Parameters
        ----------
        filename : str
            Path to the output file.
        """
        with self.io(filename, "a") as fp:
            fp.write_samples(self.samples)
