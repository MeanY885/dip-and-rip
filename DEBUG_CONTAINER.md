# Docker Container Debug Instructions

The UI changes aren't showing up despite rebuilds. Here's how to debug:

## 1. Update and Rebuild Container

First, ensure you have the latest code:

```bash
git pull
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## 2. Check Debug Endpoint

Visit this URL in your browser:
```
http://10.90.90.16:8000/api/debug/template-deployment
```

This will show:
- Whether the template file exists in the container
- File hash and modification time
- Whether the new features are present
- File size and other stats

## 3. Expected Debug Output

You should see JSON like:
```json
{
  "success": true,
  "debug_info": {
    "template_path": "/app/templates/data_viewer.html",
    "file_exists": true,
    "features": {
      "has_data_type_dropdown": true,
      "has_swing_button": true,
      "has_swing_container": true
    }
  }
}
```

## 4. Manual Container Inspection

If the debug endpoint shows issues, inspect the container directly:

```bash
# Get container name
docker ps

# Execute shell in container
docker exec -it <container_name> /bin/bash

# Check template file
cat /app/templates/data_viewer.html | grep -A5 "Data Type"
cat /app/templates/data_viewer.html | grep "loadSwingAnalysis"
```

## 5. Browser Cache Issues

Even after clearing cache, try:
- Hard refresh: Ctrl+F5 (Windows/Linux) or Cmd+Shift+R (Mac)
- Open in incognito/private browsing mode
- Try different browser

## 6. Container Logs

Check for errors:
```bash
docker-compose logs -f
```

## What to Look For

The new UI should have:
1. **Data Type dropdown** with "Daily Data" and "Minute Data" options
2. **Load Swing Analysis button** (blue button)
3. Both elements should be visible on the Data Viewer page

## If Still Not Working

1. Check the debug endpoint output
2. Share the container logs
3. Confirm the template file hash matches the local version
4. Try accessing from a different device on the network

The template file in your local repo has MD5 hash: `0a3880a8000e747602fcea6b95cd8948` - the container should match this.