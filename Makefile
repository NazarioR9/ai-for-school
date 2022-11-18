init: build start install

start:
	@docker-compose up -d

stop:
	@docker-compose stop

restart: stop start

build:
	@docker-compose build
	
install:
	@docker-compose exec app pip install .

clean:
	@docker-compose down

pull:
	@git checkout .
	@git pull

remove-cache:
	@docker-compose exec app rm -rf /home/appuser/.cache/*
	
run-test:
	@docker-compose exec app python -m pytest
