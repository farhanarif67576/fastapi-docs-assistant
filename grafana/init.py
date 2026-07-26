"""
Grafana provisioning script.
Creates the PostgreSQL data source and imports the monitoring dashboard
via the Grafana HTTP API.

Usage:
    # Make sure Grafana is running at http://localhost:3000
    uv run python grafana/init.py

Environment variables:
    GRAFANA_URL: Grafana URL (default: http://localhost:3000)
    GRAFANA_USER: Grafana admin user (default: admin)
    GRAFANA_PASSWORD: Grafana admin password (default: admin)
"""

import os
import json
import requests
from typing import Dict, Any


# ── Configuration ──────────────────────────────────────────────────────────────

GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
GRAFANA_USER = os.getenv("GRAFANA_USER", "admin")
GRAFANA_PASSWORD = os.getenv("GRAFANA_PASSWORD", "admin")

# Dashboard file path
DASHBOARD_PATH = os.path.join(os.path.dirname(__file__), "dashboard.json")

# PostgreSQL data source configuration
POSTGRES_DATASOURCE = {
    "name": "PostgreSQL",
    "type": "postgres",
    "access": "proxy",
    "url": "postgres:5432",
    "database": "fastapi_assistant",
    "user": "user",
    "secureJsonData": {
        "password": "password",
    },
    "jsonData": {
        "sslmode": "disable",
        "postgresVersion": 1700,
        "timescaledb": False,
    },
    "isDefault": True,
}


# ── API Helper ─────────────────────────────────────────────────────────────────

def grafana_api(method: str, path: str, data: Dict[str, Any] = None) -> requests.Response:
    """Make an authenticated request to the Grafana API."""
    url = f"{GRAFANA_URL}/api{path}"
    headers = {"Content-Type": "application/json"}
    
    response = requests.request(
        method=method,
        url=url,
        headers=headers,
        auth=(GRAFANA_USER, GRAFANA_PASSWORD),
        json=data,
    )
    
    return response


# ── Provisioning Functions ─────────────────────────────────────────────────────

def create_datasource() -> bool:
    """Create or update the PostgreSQL data source."""
    print("📡 Creating PostgreSQL data source...")
    
    response = grafana_api("POST", "/datasources", POSTGRES_DATASOURCE)
    
    if response.status_code == 200:
        print("✅ Data source created successfully!")
        return True
    elif response.status_code == 409:
        print("⚠️  Data source already exists. Updating...")
        # Find existing data source
        list_response = grafana_api("GET", "/datasources")
        if list_response.status_code == 200:
            for ds in list_response.json():
                if ds["name"] == POSTGRES_DATASOURCE["name"]:
                    ds_id = ds["id"]
                    update_response = grafana_api("PUT", f"/datasources/{ds_id}", POSTGRES_DATASOURCE)
                    if update_response.status_code == 200:
                        print("✅ Data source updated!")
                        return True
        print("❌ Could not update data source")
        return False
    else:
        print(f"❌ Failed to create data source: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
        return False


def import_dashboard() -> bool:
    """Import the dashboard JSON file."""
    print("\n📊 Importing dashboard...")
    
    # Load dashboard JSON
    try:
        with open(DASHBOARD_PATH, "r", encoding="utf-8") as f:
            dashboard_json = json.load(f)
    except FileNotFoundError:
        print(f"❌ Dashboard file not found: {DASHBOARD_PATH}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Invalid dashboard JSON: {e}")
        return False
    
    # Prepare the import payload
    payload = {
        "dashboard": dashboard_json,
        "overwrite": True,
        "message": "Provisioned by init.py",
    }
    
    response = grafana_api("POST", "/dashboards/db", payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Dashboard imported: {result.get('url', 'unknown')}")
        return True
    else:
        print(f"❌ Failed to import dashboard: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
        return False


def check_health() -> bool:
    """Check if Grafana is accessible."""
    try:
        response = requests.get(
            f"{GRAFANA_URL}/api/health",
            timeout=5,
        )
        if response.status_code == 200:
            version = response.json().get("version", "unknown")
            print(f"✅ Grafana is running (version {version})")
            return True
        else:
            print(f"❌ Grafana returned status {response.status_code}")
            return False
    except requests.ConnectionError:
        print(f"❌ Cannot connect to Grafana at {GRAFANA_URL}")
        print("   Make sure Grafana is running: docker compose up -d")
        return False


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    """Run the full Grafana provisioning."""
    print("=" * 60)
    print("📊 Grafana Provisioning Script")
    print("=" * 60)
    print(f"   URL:      {GRAFANA_URL}")
    print(f"   User:     {GRAFANA_USER}")
    print(f"   Dashboard: {DASHBOARD_PATH}")
    print("=" * 60)
    
    # Step 1: Check health
    print("\n🩺 Step 1: Checking Grafana health...")
    if not check_health():
        print("\n❌ Aborting provisioning.")
        return
    
    # Step 2: Create data source
    print("\n🔌 Step 2: Creating PostgreSQL data source...")
    if not create_datasource():
        print("\n⚠️  Continuing despite data source issue...")
    
    # Step 3: Import dashboard
    print("\n📈 Step 3: Importing dashboard...")
    if not import_dashboard():
        print("\n⚠️  Dashboard import had issues.")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Provisioning complete!")
    print("=" * 60)
    print(f"   Grafana: {GRAFANA_URL}")
    print(f"   Login:   {GRAFANA_USER} / {GRAFANA_PASSWORD}")
    print(f"   Dashboard: {GRAFANA_URL}/d/fastapi-assistant")
    print("=" * 60)


if __name__ == "__main__":
    main()