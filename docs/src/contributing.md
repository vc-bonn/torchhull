# Contributing

To develop your code contribution, clone the repository and install a local version of torchhull in your Python environment:

```sh
python -m pip install --editable ".[dev]"
```

We recommend to use *editable* mode for installable so your changes will become immediately visible.


## Commands

torchhull uses `nox` to handle common tasks like running code formatters, linters, documentation creation, unit testing, etc.

To execute a specific task, run the following commands in the root directory of the cloned repository:

- **Code Formatting**

    ```sh
    nox -s format
    ```

- **Code Linting**

    ```sh
    nox -s lint
    ```

- **Unit Tests**

    ```sh
    nox -s tests
    ```

- **Benchmarks**

    ```sh
    nox -s benchmarks
    ```

- **Documentation Creation**

    ```sh
    nox -s docs
    ```

- **Documentation Creation with Live Update**

    ```sh
    nox -s docs_live
    ```
