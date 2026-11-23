# BindCraft2 image, derived from the upstream containers/Dockerfile with two changes:
#
#   1. The accelerator is a build argument rather than hardcoded cuda13. Upstream's install.sh
#      reads the driver and picks; an image has to be told. Driver 550.x serves CUDA 12.4, so
#      cuda12 is correct for the vyplab nodes. CUDA 12 wheels also run on a CUDA 13 driver
#      (drivers are backward compatible), so one cuda12 image covers a mixed fleet; the reverse
#      is not true.
#
#   2. BC2 is cloned at a pinned revision instead of copied from the build context, so the
#      image does not depend on the state of a working tree and campaign_metadata.json records
#      a revision you can go back to.
#
# AlphaFold parameters are deliberately NOT baked: they carry their own licence terms, and this
# image is published. Fetch them once on the cluster into a bound directory instead.

ARG ACCELERATOR=cuda12
ARG BINDCRAFT_REF=main

FROM ubuntu:24.04
ARG ACCELERATOR
ARG BINDCRAFT_REF

# build-essential and python3-dev stay in the image: biotraj, which biotite pulls in to read
# trajectories, publishes no aarch64 wheel, and removing the compiler afterwards would pull it
# out from under whatever else links against it.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev build-essential ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_BREAK_SYSTEM_PACKAGES=1 PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

# Not --depth 1: a shallow clone cannot check out an arbitrary commit, and BINDCRAFT_REF is
# meant to accept one.
RUN git clone https://github.com/PacesaLab/BindCraft2.git /opt/bindcraft \
    && git -C /opt/bindcraft checkout --quiet "${BINDCRAFT_REF}" \
    && git -C /opt/bindcraft rev-parse HEAD > /opt/bindcraft-revision

# Editable, as upstream does it: settings/ and scaffolds/ live at the repository root rather
# than inside the package and the runtime resolves them relative to it, so the tree has to stay
# in place and on the import path. A plain install copies the package and orphans them.
RUN python3 -m pip install -e "/opt/bindcraft[${ACCELERATOR}]"

# Two things have to line up for a campaign to reach the cards, and JAX announces neither: it
# warns once and then runs on the CPU at roughly a hundredth the speed.
#   - the cuda wheels keep their libraries under nvidia/*/lib inside site-packages, where the
#     dynamic loader does not look;
#   - the NVIDIA container hook injects the host driver only into an image that asks for it by
#     these two variables, which a CUDA base image would set and plain Ubuntu does not.
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN python3 -c "import nvidia, pathlib; print('\n'.join(sorted(str(p) for r in nvidia.__path__ for p in pathlib.Path(r).glob('*/lib'))))" \
      > /etc/ld.so.conf.d/bindcraft-cuda.conf \
    && ldconfig \
    && ldconfig -p | grep -q libcupti

# Fail the build here rather than on a first campaign. pip, conda and uv all report success for
# an environment missing a module; selfcheck names every module and checkpoint. Run from / so
# what is tested is the installed package and not the source tree beside it.
WORKDIR /
RUN python3 -c "import jax, bindcraft.proteinmpnn; print('jax', jax.__version__)" \
    && bindcraft --help > /dev/null \
    && python3 -m bindcraft.selfcheck "${ACCELERATOR}" --shipped-only

LABEL org.opencontainers.image.source=https://github.com/PacesaLab/BindCraft2
LABEL org.opencontainers.image.description="BindCraft2, ${ACCELERATOR} wheels, AlphaFold parameters not included"

WORKDIR /work
ENTRYPOINT ["bindcraft"]
CMD ["--help"]
