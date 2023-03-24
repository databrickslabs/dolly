#!/bin/bash

set -e

pytest \
--cov=dolly \
--cov-report=html \
--cov-report=term \
--cov-branch \
$@