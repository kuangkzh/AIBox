# AIBox


# Run

linux:

```shell script
./setup.sh
```

windows:

```shell script
./startup.bat
```


# Debug Run

Debug mode on windows cmd:

```shell script
set FLASK_APP=Web/main.py

set FLASK_ENV=development

set FLASK_DEBUG=1

python -m flask run --host 0.0.0.0 --port 19320
```

Debug mode on windows powershell:

```shell script
$env:FLASK_APP="Web/main.py"

$env:FLASK_ENV="development"

$env:FLASK_DEBUG="1"

python -m flask run --host 0.0.0.0 --port 19320
```



Debug mode on linux:

```shell script

export FLASK_APP=Web/main.py

export FLASK_ENV=development

export FLASK_DEBUG=1

python -m flask run --host 0.0.0.0 --port 19320
```
