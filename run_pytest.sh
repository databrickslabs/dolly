#!/bin/bash

set -e

python -m pytest \
--cov=dolly \
--cov-report=html \
--cov-report=term \
--cov-branch \
$@