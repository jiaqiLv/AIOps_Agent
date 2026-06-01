"""Custom HTTP routes for LangGraph dev (topology graph viewer)."""

import asyncio
import os
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from app.utils.logger import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STATIC_TOPOLOGY = Path(__file__).resolve().parent / "static" / "topology"

LATEST_GRAPH_FILE = "propagation_graph_latest.html"
LATEST_EMBED_FILE = "propagation_graph_embed.html"
LATEST_GRAPH_PNG = "propagation_graph_latest.png"
LATEST_REPORT_FILE = "report_latest.html"


def resolve_graphs_dir() -> Path:
    """Absolute path to graph output directory (works regardless of process cwd)."""
    raw = Path(os.getenv("GRAPH_OUTPUT_DIR", "outputs/graphs"))
    graphs_dir = raw if raw.is_absolute() else _PROJECT_ROOT / raw
    graphs_dir.mkdir(parents=True, exist_ok=True)
    return graphs_dir


GRAPHS_DIR = resolve_graphs_dir()


def resolve_reports_dir() -> Path:
    """Absolute path to report output directory."""
    raw = Path(os.getenv("REPORT_OUTPUT_DIR", "outputs/reports"))
    d = raw if raw.is_absolute() else _PROJECT_ROOT / raw
    d.mkdir(parents=True, exist_ok=True)
    return d


REPORTS_DIR = resolve_reports_dir()


async def latest_topology(_request):
    """Serve the most recently generated propagation topology HTML."""
    latest = GRAPHS_DIR / LATEST_GRAPH_FILE
    if not await asyncio.to_thread(latest.is_file):
        return JSONResponse(
            {
                "error": "no_graph_yet",
                "message": "尚未生成拓扑图。请先运行一次诊断（KE-FPC 分析完成后会自动生成）。",
                "graphs_dir": str(GRAPHS_DIR.resolve()),
            },
            status_code=404,
        )
    return FileResponse(
        path=str(latest),
        media_type="text/html; charset=utf-8",
        content_disposition_type="inline",
    )


async def latest_topology_png(_request):
    """Serve latest PNG for inline display in Studio chat."""
    latest = GRAPHS_DIR / LATEST_GRAPH_PNG
    if not await asyncio.to_thread(latest.is_file):
        return JSONResponse(
            {"error": "no_graph_yet", "message": "尚未生成拓扑 PNG，请先完成一次诊断。"},
            status_code=404,
        )
    return FileResponse(
        path=str(latest),
        media_type="image/png",
        content_disposition_type="inline",
    )


async def topology_index(_request):
    return RedirectResponse(url="/topology/latest", status_code=302)


async def topology_embed(_request):
    """Fullscreen draggable topology (large node labels)."""
    latest = GRAPHS_DIR / LATEST_EMBED_FILE
    if not await asyncio.to_thread(latest.is_file):
        return RedirectResponse(url="/topology/latest", status_code=302)
    return FileResponse(
        path=str(latest),
        media_type="text/html; charset=utf-8",
        content_disposition_type="inline",
    )


async def topology_interactive(_request):
    return RedirectResponse(url="/topology/embed", status_code=302)


async def latest_report(_request):
    """Serve the most recently generated HTML report."""
    latest = REPORTS_DIR / LATEST_REPORT_FILE
    if not await asyncio.to_thread(latest.is_file):
        return JSONResponse(
            {
                "error": "no_report_yet",
                "message": "尚未生成报告。请先完成一次根因分析。",
                "reports_dir": str(REPORTS_DIR.resolve()),
            },
            status_code=404,
        )
    return FileResponse(
        path=str(latest),
        media_type="text/html; charset=utf-8",
        content_disposition_type="inline",
    )


def get_report_url() -> str:
    """Public URL for the latest report page."""
    base = os.getenv("LANGGRAPH_PUBLIC_BASE_URL", "http://192.168.199.5:32024").rstrip("/")
    return f"{base}/report/latest"


app = Starlette(
    routes=[
        Route("/topology/latest", endpoint=latest_topology, methods=["GET"]),
        Route("/topology/embed", endpoint=topology_embed, methods=["GET"]),
        Route("/topology/latest.png", endpoint=latest_topology_png, methods=["GET"]),
        Route("/topology/interactive", endpoint=topology_interactive, methods=["GET"]),
        Route("/topology", endpoint=topology_index, methods=["GET"]),
        Route("/report/latest", endpoint=latest_report, methods=["GET"]),
        Mount(
            "/topology/static",
            app=StaticFiles(directory=str(_STATIC_TOPOLOGY)),
            name="topology_static",
        ),
        Mount(
            "/topology/files",
            app=StaticFiles(directory=str(GRAPHS_DIR), html=True),
            name="topology_files",
        ),
        Mount(
            "/report/files",
            app=StaticFiles(directory=str(REPORTS_DIR), html=True),
            name="report_files",
        ),
    ],
)

logger.info(
    "Topology viewer at /topology/latest, report at /report/latest "
    "(static JS at /topology/static/, files at /topology/files/)"
)
