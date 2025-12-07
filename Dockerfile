FROM astral/uv:python3.12-bookworm-slim AS base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y upgrade && apt-get install -y libavcodec-dev ffmpeg


FROM base AS build

RUN mkdir -p /app/build
COPY src README.md pyproject.toml /app/build/

WORKDIR /app/build
RUN uv build


FROM base AS runtime

RUN mkdir -p /tmp/wheels
COPY --from=build /app/build/dist/*.whl /tmp/wheels/
RUN uv tool install /tmp/wheels/*.whl
RUN rm -rf /tmp/wheels

WORKDIR /app
RUN mkdir audio_prompts

EXPOSE 8004

ENTRYPOINT [ "uv", "run", "echo-tts-api" ]

