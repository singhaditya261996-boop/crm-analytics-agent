.PHONY: start start-bg stop restart logs build reset update shell check switch-small network status

## Start the agent (foreground — see live logs)
start:
	docker-compose up

## Start the agent in background (silent mode)
start-bg:
	docker-compose up -d
	@echo ""
	@echo "✅ Agent running in background"
	@echo "   Open: http://localhost:8501"
	@echo "   Logs: make logs"
	@echo "   Stop: make stop"
	@echo ""

## Stop the agent
stop:
	docker-compose down
	@echo "✅ Agent stopped"

## Restart the agent
restart:
	docker-compose restart
	@echo "✅ Agent restarted — http://localhost:8501"

## View live logs
logs:
	docker-compose logs -f

## Build the Docker image (run after code changes)
build:
	docker-compose build
	@echo "✅ Build complete — run 'make start' to launch"

## Full reset — deletes cached models (will re-download on next start)
reset:
	@echo "⚠️  WARNING: This deletes all cached AI models (9GB+)"
	@echo "   They will re-download on next start (takes 30+ mins)"
	@read -p "   Are you sure? Type 'yes' to confirm: " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		docker-compose down -v; \
		echo "✅ Reset complete"; \
	else \
		echo "Cancelled"; \
	fi

## Update to latest code (rebuilds image)
update:
	git pull
	docker-compose up --build

## Open a shell inside the running container (for debugging)
shell:
	docker exec -it crm-analytics-agent bash

## Run pre-flight checks before first launch
check:
	@bash check_requirements.sh

## Switch to smaller faster model (for low-RAM machines)
switch-small:
	docker exec crm-analytics-agent ollama pull llama3.2:3b
	@echo ""
	@echo "✅ Small model downloaded"
	@echo "   Update config/settings.yaml: set model to llama3.2:3b"
	@echo "   Then run: make restart"
	@echo ""

## Share with teammates on same WiFi network
network:
	@echo ""
	@echo "🌐 Starting in network mode — teammates can connect on same WiFi"
	@IP=$$(hostname -I 2>/dev/null | awk '{print $$1}' || ipconfig getifaddr en0 2>/dev/null || echo "your-ip"); \
	echo "   Send teammates this link: http://$$IP:8501"; \
	echo ""
	docker-compose up

## Show container status
status:
	@docker ps --filter name=crm-analytics-agent \
		--format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
