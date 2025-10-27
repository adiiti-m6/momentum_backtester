# Streamlit Deployment Guide

## Deploy to Streamlit Community Cloud (Recommended)

Streamlit Community Cloud is the **easiest way** to deploy this app — it's built specifically for Streamlit.

### Quick Deploy Steps

1. **Fork or ensure your repo is on GitHub**: Your repo `adiiti-m6/momentum_backtester` is already connected.

2. **Go to Streamlit Community Cloud**:
   - Visit: https://streamlit.io/cloud
   - Click **"New app"**

3. **Connect your repository**:
   - Select your GitHub repo: `adiiti-m6/momentum_backtester`
   - Select branch: `master`
   - Set main file path: `src/app/streamlit_app.py`

4. **Deploy**:
   - Click **"Deploy"**
   - Wait ~2-3 minutes for build and startup
   - Your app will be live at: `https://<your-custom-url>.streamlit.app`

### Configuration Files Already Added

- **`.streamlit/config.toml`** — Streamlit settings (theme, server config, max upload size 200MB)
- **`requirements.txt`** — All Python dependencies pinned

### Troubleshooting

**If deployment fails:**
- Check the **Logs** tab in Streamlit Cloud dashboard
- Verify `requirements.txt` has all dependencies
- Ensure `src/app/streamlit_app.py` is the correct entry point

**If app is slow:**
- Add caching decorators (`@st.cache_data`) to expensive operations (already done in your code)
- Check data file sizes (CSV uploads should be < 200MB by default)

**If you see import errors:**
- Ensure `pyproject.toml` has `packages = [{include = "src"}]`
- Verify all imports use relative paths from `src`

---

## Alternative Deployments

### Render (Docker-based)

If you prefer Render over Streamlit Cloud:
1. A `Dockerfile` and `render.yaml` can be added for Docker deployment
2. Render will auto-build and run the Docker image
3. Set environment variable: `STREAMLIT_SERVER_PORT=8501`

### Fly.io or Railway

Similar Docker-based setup available. Contact me if you'd like a Dockerfile.

### Local Testing Before Deploy

```bash
# Activate environment
.\.venv\Scripts\Activate.ps1

# Run locally
streamlit run src/app/streamlit_app.py

# Should open browser at http://localhost:8501
```

---

## Next Steps

1. Go to https://streamlit.io/cloud and sign in with your GitHub account
2. Click **"New app"** → select your repo → deploy
3. Your live app will be ready in ~2-3 minutes!

For support, see [Streamlit Docs](https://docs.streamlit.io/deploy/streamlit-community-cloud).
