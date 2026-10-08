.PHONY: up down reset psql test lint
up:
	docker compose up -d
down:
	docker compose down
reset:            # destroys the database volume and re-runs migrations
	docker compose down -v && docker compose up -d
psql:
	docker compose exec db sh -c 'psql -U $$POSTGRES_USER -d $$POSTGRES_DB'
test:
	pytest
lint:
	ruff check .
