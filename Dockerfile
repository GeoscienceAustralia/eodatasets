
# This gdal version should match the "gdal=="
# line in setup.py's "docker"
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.6.3

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHONFAULTHANDLER=1

# Apt installation
RUN apt-get update && \
    apt-get install -y \
      build-essential \
      git \
      vim \
      nano \
      tini \
      wget \
      python3-pip \
      # For Psycopg2
      libpq-dev python3-dev \
    && apt-get autoclean && \
    apt-get autoremove && \
    rm -rf /var/lib/{apt,dpkg,cache,log}


# Environment can be whatever is supported by setup.py
# so, either deployment, test
ARG ENVIRONMENT=test
# ARG ENVIRONMENT=deployment

RUN echo "Environment is: $ENVIRONMENT"


# Pip installation
RUN mkdir -p /conf
COPY requirements /conf/

RUN pip install -r /conf/setup.txt

ENV PATH=/usr/local/bin:$PATH

RUN pip install -r /conf/${ENVIRONMENT}.txt

# USER runner ?

# Dev setup: run pre-commit once, so its virtualenv is built and cached.
#    We do this in a tmp repository, before copying our real code, as we
#    want this cached by Docker and not rebuilt every time code changes
COPY .pre-commit-config.yaml /conf/

RUN if [ "$ENVIRONMENT" = "test" ] ; then \
       mkdir -p ~/pre-commit \
       && cp /conf/.pre-commit-config.yaml ~/pre-commit \
       && cd ~/pre-commit \
       && git init \
       && pre-commit run \
       && rm -rf ~/pre-commit ; \
    fi


# Set up a nice workdir and add the live code
ENV APPDIR=/code
RUN mkdir -p $APPDIR
WORKDIR $APPDIR
ADD . $APPDIR

# These ENVIRONMENT flags make this a bit complex, but basically, if we are in dev
# then we want to link the source (with the -e flag) and if we're in prod, we
# want to delete the stuff in the /code folder to keep it simple.
#
# (note: --editable doesn't currently work well with pyproject.toml projects, so
#        we turn off pep517 with it)
#
RUN if [ "$ENVIRONMENT" = "deployment" ] ; then\
        pip install .[$ENVIRONMENT] ; \
        rm -rf /code/* ; \
    else \
        pip install --no-use-pep517 --editable .[$ENVIRONMENT] ; \
    fi

RUN pip freeze

# Is it working?
RUN eo3-validate --version

ENTRYPOINT ["/bin/tini", "--"]
CMD ["eo3-validate"]
