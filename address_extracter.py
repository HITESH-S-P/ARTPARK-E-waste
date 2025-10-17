import os
import io
import time
import json
import requests
import pandas as pd
import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from datetime import datetime

# -------------------------
# App config & styling
# -------------------------
st.set_page_config(
    page_title="PDF Address Extractor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --dark-green: #2c5234;
        --medium-green: #4c7c5c;
        --light-green: #e8f0ec;
        --light-beige: #f9f6f1;
        --accent-blue: #2c5985;
        --text-dark: #2d2d2d;
        --text-light: #ffffff;
    }

    /* ---------- MAIN BACKGROUND & TEXT ---------- */
    .main, .stApp {
        background-color: var(--light-beige) !important;
        color: var(--text-dark) !important;
    }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"] {
        background-color: var(--medium-green) !important;
        color: var(--text-light) !important;
    }

    /* ---------- HEADINGS ---------- */
    h1, h2, h3 {
        color: var(--dark-green) !important;
        font-family: 'Arial', sans-serif;
        margin-bottom: 1rem !important;
        font-weight: 700;
    }

    /* ---------- PARAGRAPH TEXT ---------- */
    p, .stText, .stMarkdown {
        color: var(--text-dark) !important;
    }

    /* ---------- BUTTONS ---------- */
    .stButton > button {
        background-color: var(--medium-green) !important;
        color: var(--text-light) !important;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
        border: none;
    }
    .stButton > button:hover {
        background-color: var(--dark-green) !important;
        transform: translateY(-1px);
    }

    /* ---------- TEXT INPUTS & TEXT AREAS ---------- */
    .stTextArea textarea, .stTextInput input {
        background-color: #1f2937 !important;
        color: var(--text-light) !important;
        -webkit-text-fill-color: var(--text-light) !important;
        caret-color: var(--text-light) !important;
        border: 1px solid var(--medium-green) !important;
        border-radius: 6px;
        padding: 0.6rem;
        font-family: 'Consolas', monospace;
    }

    /* ---------- DATAFRAMES ---------- */
    .stDataFrame {
        background-color: var(--light-green);
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }

    /* ---------- DOWNLOAD BUTTONS ---------- */
    .stDownloadButton > button {
        background-color: var(--accent-blue);
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
        border: none;
    }
    .stDownloadButton > button:hover {
        background-color: #1e3d5c;
        transform: translateY(-1px);
    }

    /* ---------- MAP CONTAINERS ---------- */
    div[data-testid="stVerticalBlock"] > div:has(div.stMap) {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        height: 600px !important;
        min-height: 600px !important;
    }
    .folium-map, .stMap {
        border-radius: 8px;
        overflow: hidden;
        height: 600px !important;
        background-color: white !important;
    }

    /* ---------- ALERTS & PROGRESS ---------- */
    .stAlert {
        border-radius: 8px;
        padding: 1rem;
    }
    .stProgress > div > div {
        background-color: var(--medium-green) !important;
    }

    /* ---------- CUSTOM CONTAINER ---------- */
    .custom-container {
        background-color: white;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }

    /* ---------- FOOTER SIGNATURE ---------- */
    .footer {
        position: fixed;
        bottom: 10px;
        right: 15px;
        background: linear-gradient(90deg, #2c5234, #4c7c5c, #2c5985);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        font-size: 15px;
        letter-spacing: 0.5px;
        animation: fadeIn 3s ease-in-out;
        text-shadow: 0px 0px 12px rgba(76, 124, 92, 0.3);
        user-select: none;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0px); }
    }
</style>

<!-- Signature Footer -->
<div class="footer">Developed by <strong>Aneesh</strong></div>
""", unsafe_allow_html=True)




GENAI_API_KEY="AIzaSyBDeIVUULJoaiu1SBmW5vBQI1h3lCCP3rw"
# AIzaSyDwQCx1HS0yZuDmQjgeiV03VPl27r_kkOc
GOOGLE_MAPS_API_KEY="AIzaSyCud3zIRNArlw2aoxgyDx7l1tMDaDJtBug"


# Configure Generative AI
if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)

GEMINI_MODEL_ID = "gemini-2.5-flash"

if "parsed_data" not in st.session_state:
    st.session_state.parsed_data = None
    st.session_state.raw_output = ""
    st.session_state.edited_data = None
    st.session_state.uploaded_filename = None

def clear_session_state():
    st.session_state.parsed_data = None
    st.session_state.raw_output = ""
    st.session_state.edited_data = None
    st.session_state.uploaded_filename = None

# -------------------------
# Utility conversions
# -------------------------
def convert_to_csv(data):
    df = pd.DataFrame(data)
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df.to_csv(index=False).encode("utf-8")

def convert_to_json(data):
    def _default(o):
        if isinstance(o, datetime):
            return o.strftime('%Y-%m-%d %H:%M:%S')
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")
    return json.dumps(data, indent=2, default=_default).encode("utf-8")

# Robust JSON cleanup
def clean_json_output(raw_output: str):
    try:
        s = raw_output.strip().replace("```json", "").replace("```", "").strip()
        # Try direct parse
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                return [obj]
        except json.JSONDecodeError:
            pass
        # Try bracket slicing
        start, end = s.find("["), s.rfind("]") + 1
        if start >= 0 and end > start:
            sliced = s[start:end]
            return json.loads(sliced)
    except Exception:
        pass
    return []

def geocode_address(address: str, bias_city="Bangalore, Karnataka, India", retries=2, backoff=0.8):
    """Return dict {latitude, longitude, formatted_address} or None."""
    full_query = f"{address}, {bias_city}" if bias_city and bias_city.lower() not in address.lower() else address
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": full_query, "key": GOOGLE_MAPS_API_KEY}

    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=15)
            data = r.json()
            status = data.get("status", "UNKNOWN_ERROR")
            if status == "OK" and data.get("results"):
                loc = data["results"][0]["geometry"]["location"]
                return {
                    "complete_address": data["results"][0]["formatted_address"],
                    "latitude": float(loc["lat"]),
                    "longitude": float(loc["lng"]),
                }
            elif status in ("OVER_QUERY_LIMIT", "RESOURCE_EXHAUSTED", "UNKNOWN_ERROR"):
                time.sleep(backoff * (attempt + 1))
                continue
            else:
                return None
        except Exception:
            time.sleep(backoff * (attempt + 1))
    return None

def extract_addresses_with_gemini(text: str):
    if not GENAI_API_KEY:
        st.error("Missing Generative AI API key.")
        return []

    prompt = f"""
Extract postal-style address strings from the text and return a strict JSON array.
Each item must be an object with exactly one field:
  "complete_address": "<full address line>"
Rules:
1) Include only addresses in Bengaluru (Bangalore), Karnataka, India. Discard others.
2) Do not include coordinates; only the full address string.
3) Return a valid JSON array even if empty, e.g., [].
4) Do not include comments or markdown fences.

TEXT:
\"\"\"{text}\"\"\"
"""

    try:
        model = genai.GenerativeModel(GEMINI_MODEL_ID)
        resp = model.generate_content(prompt)
        raw = getattr(resp, "text", "") or ""
        parsed = clean_json_output(raw)
        # Ensure shape: list of dicts with 'complete_address'
        cleaned = []
        for item in parsed:
            if isinstance(item, dict) and "complete_address" in item and isinstance(item["complete_address"], str):
                addr = item["complete_address"].strip()
                if addr:
                    cleaned.append({"complete_address": addr})
            elif isinstance(item, str):
                s = item.strip()
                if s:
                    cleaned.append({"complete_address": s})
        return cleaned, raw
    except Exception as e:
        st.error(f"LLM extraction failed: {e}")
        return [], ""

# ---------------------------------
# UI
# ---------------------------------
st.markdown('<div class="custom-container">', unsafe_allow_html=True)
st.title("PDF Address Extractor & Mapper")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    st.session_state.uploaded_filename = uploaded_file.name

    # Extract text with progress bar
    try:
        reader = PdfReader(uploaded_file)
        total_pages = len(reader.pages)
        progress_bar = st.progress(0)
        status_text = st.empty()

        pdf_text = []
        for i, page in enumerate(reader.pages):
            status_text.text(f"Processing page {i+1}/{total_pages}")
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            if txt:
                pdf_text.append(txt)
            progress_bar.progress((i + 1) / max(total_pages, 1))

        pdf_text = "\n".join(pdf_text).strip()
        if not pdf_text:
            st.error("No extractable text found in the PDF. It may be scanned or encrypted.")
            st.stop()

        st.subheader("Extracted PDF Text")
        st.text_area("Extracted Text", pdf_text, height=250)

        if st.button("Extract Addresses and Plot") or st.session_state.parsed_data:
            if st.session_state.parsed_data is None:
                with st.spinner("Processing addresses..."):
                    addresses, raw = extract_addresses_with_gemini(pdf_text)
                    st.session_state.raw_output = raw

                    if not addresses:
                        st.error("No Bengaluru addresses detected in the text.")
                        st.stop()

                    # Geocode with a progress bar
                    g_prog = st.progress(0)
                    g_status = st.empty()
                    geocoded = []
                    for i, item in enumerate(addresses):
                        g_status.text(f"Geocoding {i+1}/{len(addresses)}")
                        addr = item.get("complete_address")
                        if addr and GOOGLE_MAPS_API_KEY:
                            res = geocode_address(addr)
                            if res:
                                geocoded.append(res)
                        g_prog.progress((i + 1) / max(1, len(addresses)))
                        time.sleep(0.05)  # gentle pacing

                    # Deduplicate (address, lat, lon)
                    seen = set()
                    unique_rows = []
                    for r in geocoded:
                        key = (r["complete_address"], r["latitude"], r["longitude"])
                        if key not in seen:
                            seen.add(key)
                            unique_rows.append(r)

                    if not unique_rows:
                        st.error("No valid geocoded results. Check API key and address quality.")
                        st.stop()

                    st.session_state.parsed_data = unique_rows
                    st.success(f"Processed {len(unique_rows)} addresses.")

    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        st.stop()

    # Optional: show raw LLM output
    if st.session_state.raw_output and st.checkbox("Show raw LLM output"):
        st.subheader("Raw JSON from LLM")
        st.code(st.session_state.raw_output, language="json")

    # Data editor + downloads + map
    if st.session_state.parsed_data:
        st.subheader("Parsed Address Data")
        df = pd.DataFrame(st.session_state.parsed_data)
        edited_df = st.data_editor(df, num_rows="dynamic")
        st.session_state.edited_data = edited_df.to_dict("records")

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download CSV",
                data=convert_to_csv(st.session_state.edited_data),
                file_name="addresses.csv",
                mime="text/csv",
            )
        with c2:
            st.download_button(
                "Download JSON",
                data=convert_to_json(st.session_state.edited_data),
                file_name="addresses.json",
                mime="application/json",
            )

        if st.button("Clear Results"):
            clear_session_state()
            st.rerun()

        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        st.subheader("Map with Plotted Addresses")

        m = folium.Map(
            location=(12.9716, 77.5946),
            zoom_start=12,
            tiles="CartoDB positron",
            height=600,
            width="100%",
        )

        marker_cluster = MarkerCluster(
            options={
                "disableClusteringAtZoom": 15,
                "maxClusterRadius": 50,
                "spiderfyOnMaxZoom": True,
            }
        ).add_to(m)

        valid = 0
        for row in st.session_state.edited_data:
            addr = row.get("complete_address")
            lat = row.get("latitude")
            lon = row.get("longitude")
            try:
                lat = float(lat)
                lon = float(lon)
                # Rough Bengaluru bounds
                if addr and 12.5 <= lat <= 13.5 and 77.0 <= lon <= 78.0:
                    folium.Marker(
                        location=(lat, lon),
                        popup=addr,
                        tooltip=addr,
                        icon=folium.Icon(color="green", icon="info-sign"),
                    ).add_to(marker_cluster)
                    valid += 1
                else:
                    st.warning(f"Coordinates outside Bengaluru bounds: {addr} (lat={lat}, lon={lon})")
            except (TypeError, ValueError):
                st.warning(f"Invalid coordinates for: {addr}")

        # Export as HTML
        map_html = m._repr_html_()
        st.download_button(
            "Download Map HTML",
            data=map_html,
            file_name="address_map.html",
            mime="text/html",
        )

        if valid > 0:
            st_folium(m, width="100%", height=600, returned_objects=[])
            st.success(f"Plotted {valid} valid locations.")
        else:
            st.info("No valid locations to plot.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Upload a PDF and click 'Extract Addresses and Plot' to begin.")
else:
    st.info("Please upload a PDF to start.")
st.markdown("</div>", unsafe_allow_html=True)