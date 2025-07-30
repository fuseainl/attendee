#!/bin/bash
# entrypoint.sh: wacht tot Postgres start, doe migraties, start daarna de server

set -e

echo "Waiting for Postgres to be ready..."
until pg_isready -h localhost -p 5432; do
  echo "Postgres is unavailable - sleeping"
  sleep 2
done

echo "Postgres is up - running migrations"
python manage.py migrate

echo "Starting command: $*"
exec "$@"