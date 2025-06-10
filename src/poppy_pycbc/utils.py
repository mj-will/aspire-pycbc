import ast
from collections import namedtuple
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
import numpy.lib.recfunctions as rfn
from poppy.samples import Samples

if TYPE_CHECKING:
    import pycbc.inference.model


Inputs = namedtuple(
    "Inputs",
    [
        "log_likelihood",
        "log_prior",
        "dims",
        "parameters",
        "prior_bounds",
        "periodic_parameters",
    ],
)
"""Container for the inputs to the poppy sampler."""


Functions = namedtuple("Functions", ["log_likelihood", "log_prior"])
"""Container for the log likelihood and log prior functions."""


@dataclass
class GlobalVariables:
    """Dataclass to store global functions and model reference.

    Attributes
    ----------
    pycbc_model : pycbc.inference.model.Model
        The PyCBC model object.
    loglikelihood_function : str
        Name of the loglikelihood function to use.
    """

    pycbc_model: "pycbc.inference.model.Model"
    loglikelihood_function: str = "loglikelihood"


_global_variables = GlobalVariables(None)


def update_global_variables(
    pycbc_model: "pycbc.inference.model.Model",
    loglikelihood_function: str = "loglikelihood",
) -> None:
    """Update the global variables for log likelihood and log prior.

    Parameters
    ----------
    pycbc_model : pycbc.inference.model.Model
        The PyCBC model to use.
    loglikelihood_function : str, optional
        Name of the loglikelihood function to use (default: "loglikelihood").
    """
    global _global_variables
    _global_variables.pycbc_model = pycbc_model
    _global_variables.loglikelihood_function = loglikelihood_function


def _global_log_prior(x: np.ndarray) -> float:
    """Evaluate the global log prior at a given point.

    Parameters
    ----------
    x : np.ndarray
        Parameter values.

    Returns
    -------
    float
        Log prior value.
    """
    model = _global_variables.pycbc_model
    model.update(**dict(zip(model.sampling_params, x)))
    return model.logprior


def _global_log_likelihood(x: np.ndarray) -> float:
    """Evaluate the global log likelihood at a given point.

    Parameters
    ----------
    x : np.ndarray
        Parameter values.

    Returns
    -------
    float
        Log likelihood value.
    """
    model = _global_variables.pycbc_model
    model.update(**dict(zip(model.sampling_params, x)))
    return getattr(model, _global_variables.loglikelihood_function)


def get_periodic_parameters(model: "pycbc.inference.model.Model") -> List[str]:
    """Get a list of periodic parameters from the model.

    Parameters
    ----------
    model : pycbc.inference.model.Model
        The PyCBC model.

    Returns
    -------
    list of str
        List of periodic parameter names.
    """
    periodic = []
    cyclic = model.prior_distribution.cyclic
    for param in model.variable_params:
        if param in cyclic:
            periodic.append(param)
    return periodic


def get_prior_bounds(
    model: "pycbc.inference.model.Model",
) -> Dict[str, List[float]]:
    """Get the prior bounds for the model.

    Parameters
    ----------
    model : pycbc.inference.model.Model
        The PyCBC model.

    Returns
    -------
    dict
        Dictionary mapping parameter names to [min, max] bounds.
    """
    bounds = {}
    for dist in model.prior_distribution.distributions:
        bounds.update(
            **{
                k: [v.min, v.max]
                for k, v in dist.bounds.items()
                if k in model.sampling_params
            }
        )
    return bounds


def get_poppy_functions(
    model: "pycbc.inference.model.Model",
    loglikelihood_function: str = "loglikelihood",
) -> Functions:
    """Get log likelihood and log prior functions for poppy.

    Parameters
    ----------
    model : pycbc.inference.model.Model
        The PyCBC model.
    loglikelihood_function : str, optional
        Name of the loglikelihood function to use (default: "loglikelihood").

    Returns
    -------
    Functions
        Named tuple containing log_likelihood and log_prior callables.
    """
    update_global_variables(model, loglikelihood_function)

    def log_likelihood(samples: Samples, map_fn=map) -> np.ndarray:
        """Evaluate log likelihood for a set of samples."""
        return np.fromiter(
            map_fn(_global_log_likelihood, samples.x),
            dtype=float,
        )

    def log_prior(samples: Samples, map_fn=map) -> np.ndarray:
        """Evaluate log prior for a set of samples."""
        return np.fromiter(
            map_fn(_global_log_prior, samples.x),
            dtype=float,
        )

    return Functions(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
    )


def get_inputs_from_pycbc_model(
    model: "pycbc.inference.model.Model",
    loglikelihood_function: str = "loglikelihood",
) -> Inputs:
    """Get the inputs for the poppy sampler from a PyCBC model.

    Parameters
    ----------
    model : pycbc.inference.model.Model
        The PyCBC model to extract inputs from.
    loglikelihood_function : str, optional
        Name of the loglikelihood function to use (default: "loglikelihood").

    Returns
    -------
    Inputs
        A named tuple containing the inputs for the poppy sampler.
    """
    functions = get_poppy_functions(model, loglikelihood_function)
    return Inputs(
        log_likelihood=functions.log_likelihood,
        log_prior=functions.log_prior,
        dims=len(model.sampling_params),
        parameters=list(model.sampling_params),
        prior_bounds=get_prior_bounds(model),
        periodic_parameters=get_periodic_parameters(model),
    )


def samples_from_pycbc_model(
    model: "pycbc.inference.model.Model",
    n_samples: int,
) -> Samples:
    """Draw samples from the prior distribution of a PyCBC model.

    Parameters
    ----------
    model : pycbc.inference.model.Model
        The PyCBC model.
    n_samples : int
        Number of samples to draw.

    Returns
    -------
    Samples
        Samples object containing drawn samples and parameter names.
    """
    theta = model.prior_distribution.rvs(size=n_samples)
    return Samples(
        x=rfn.structured_to_unstructured(theta[list(model.sampling_params)]),
        parameters=model.sampling_params,
    )


def samples_from_pycbc_result(
    result_file: str, parameters: List[str]
) -> Samples:
    """Get samples from a PyCBC result file.

    Parameters
    ----------
    result_file : str
        Path to the PyCBC result file.
    parameters : list of str
        List of parameter names to extract.

    Returns
    -------
    Samples
        Samples object containing extracted samples and parameter names.
    """
    from pycbc.inference.io import loadfile

    with loadfile(result_file, "r") as fp:
        samples = fp.read_samples(parameters)
    return Samples(
        x=rfn.structured_to_unstructured(samples[list(parameters)]),
        parameters=parameters,
    )


def try_literal_eval(value: str) -> Any:
    """Try to evaluate a string as a Python literal.

    Parameters
    ----------
    value : str
        The string to evaluate.

    Returns
    -------
    Any
        The evaluated value, or the original string if evaluation fails.
    """
    try:
        return ast.literal_eval(value)
    except ValueError:
        return value
