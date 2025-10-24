# aspire-pycbc

`pycbc` interface for the `aspire` sampler.


## Installation

For now, the interface has to be installed from source:

```
pip install git+https://github.com/mj-will/aspire-pycbc.git
```

and requires a custom version of PyCBC: https://github.com/mj-will/pycbc/tree/add-aspire-sampler


## Usage

### Using ini files

`aspire-pycbc` defines several custom sections in the ini file for configuring
the different methods used when calling aspire:

- `aspire.fit`: arguments passed to `Aspire.fit` when fitting the flow
- `aspire.sample`: arguments passed to `Aspire.sample` when drawing posterior samples
- `aspire.sample.sampler_kwargs` keyword arguments passed to the sampler when using for example SMC sampler
