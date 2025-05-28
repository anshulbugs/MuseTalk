#!/usr/bin/env python3
"""
Simple HTTP server to serve the JobTalk HTML page
"""

import asyncio
import logging
from pathlib import Path
from aiohttp import web
import aiohttp_cors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def serve_jobtalk_page(request):
    """Serve the JobTalk HTML page"""
    html_file = Path(__file__).parent / 'jobtalk_simple.html'
    
    if not html_file.exists():
        return web.Response(text="JobTalk HTML file not found", status=404)
    
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return web.Response(text=content, content_type='text/html')

async def main():
    app = web.Application()
    
    # Setup CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })
    
    # Add route
    app.router.add_get('/', serve_jobtalk_page)
    
    # Add CORS to all routes
    for route in list(app.router.routes()):
        cors.add(route)
    
    # Start server
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, 'localhost', 3000)
    await site.start()
    
    logger.info("JobTalk HTML server started at http://localhost:3000")
    logger.info("Open your browser and navigate to http://localhost:3000")
    logger.info("Make sure your MuseTalk server is running on port 8080")
    
    try:
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())