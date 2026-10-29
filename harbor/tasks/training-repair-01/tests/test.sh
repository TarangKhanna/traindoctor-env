#!/bin/sh
set -eu
exec python /tests/grade.py --submission /submission --out /logs/verifier --fixture /opt/traindoctor-fixture
