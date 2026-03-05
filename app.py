#!/usr/bin/env python3
"""
FoodMapper - Semantic Matching Application
USDA Agricultural Research Service, Western Human Nutrition Research Center

Matches food descriptions between databases using neural embeddings (GTE-Large model).
API-first with automatic CPU fallback for reliability.
"""

import os
import io
import json
import re
import time
import asyncio
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo
import shinyswatch
from shinywidgets import render_widget, output_widget
from functools import lru_cache

# ============================================================================
# IMPORTS
# ============================================================================

# Matching algorithms
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# API client for embeddings
from openai import OpenAI, AsyncOpenAI
import httpx
try:
    import h2  # type: ignore
    HTTP2_AVAILABLE = True
except Exception:
    HTTP2_AVAILABLE = False

# ============================================================================
# STYLE CONFIGURATION 
# ============================================================================
custom_css = """
/* Professional neutral color scheme */
:root {
    --primary-color: #475569;
    --primary-dark: #334155;
    --secondary-color: #64748b;
    --success-color: #059669;
    --warning-color: #d97706;
    --danger-color: #dc2626;
    --background: #ffffff;
    --surface: #f8fafc;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --border-color: #e2e8f0;
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.08);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
    --gradient-primary: linear-gradient(135deg, #64748b 0%, #475569 100%);
}

/* Main container with responsive design */
.container-fluid {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
    background: var(--background);
}

/* NEW: Typography with system font stack */
body, .container-fluid, .card, .btn, .table {
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    color: var(--text-primary);
    line-height: 1.6;
}

/* Clean header styling */
.app-header {
    background: var(--gradient-primary);
    color: white;
    padding: 1.25rem 1.25rem; /* NEW: thinner header */
    border-radius: 12px;
    margin-bottom: 1.25rem;
    text-align: center;
    box-shadow: var(--shadow-md);
}

.app-header h1 {
    color: white;
    margin: 0;
    font-size: 2rem; /* NEW: smaller title */
    font-weight: 700;
    letter-spacing: 0.2px;
}

.app-header p {
    color: rgba(255, 255, 255, 0.95);
    font-size: 0.95rem; /* NEW: smaller subtitle */
    margin-top: 0.35rem;
    margin-bottom: 0;
}

/* Clean card styling */
.card {
    background: white;
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow-sm);
    border-radius: 12px;
    margin-bottom: 1.5rem;
    transition: box-shadow 0.2s ease;
}

.card:hover {
    box-shadow: var(--shadow-md);
}

.card-header {
    background: linear-gradient(135deg, var(--secondary-color), var(--primary-color));
    color: white;
    font-weight: 600;
    border-radius: 12px 12px 0 0 !important;
    padding: 1rem 1.5rem;
    border-bottom: none;
}

/* Simple button styling */
.btn {
    font-weight: 600;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    transition: all 0.2s ease;
    border: none;
}

.btn-primary {
    background: var(--primary-color);
    color: white;
}

.btn-primary:hover {
    background: var(--primary-dark);
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}

.btn-success {
    background: var(--success-color);
    color: white;
}

.btn-success:hover {
    filter: brightness(0.9);
    transform: translateY(-1px);
}

.btn-warning {
    background: var(--warning-color);
    color: white;
}

.btn-warning:hover {
    filter: brightness(0.9);
    transform: translateY(-1px);
}

.btn-outline-warning {
    background: transparent;
    border: 2px solid var(--warning-color);
    color: var(--warning-color);
}

.btn-outline-warning:hover {
    background: var(--warning-color);
    color: white;
}

/* Export button grouping */
.btn-group-export {
    display: flex;
    gap: 1rem;
    justify-content: center;
    flex-wrap: wrap;
}

/* File upload area */
.file-upload-area {
    border: 2px dashed var(--border-color);
    border-radius: 12px;
    padding: 2rem;
    background: var(--background-light);
    text-align: center;
    transition: all 0.3s ease;
}

.file-upload-area:hover {
    border-color: var(--primary-color);
    background: white;
}

/* Fix table header alignment */
.shiny-table table {
    width: 100%;
    table-layout: fixed;
}

.shiny-table th {
    text-align: left !important;
    padding-left: 8px !important;
}

.shiny-table td {
    text-align: left !important;
    padding-left: 8px !important;
    word-wrap: break-word;
    overflow-wrap: break-word;
}

/* Special styling for preview tables in Step 1 - data_frame outputs */
/* Force left alignment for the entire data_frame container */
#input_col_preview,
#target_col_preview {
    text-align: left !important;
    display: block !important;
    margin: 0 !important;
    padding: 0 !important;
}

#input_col_preview .shiny-data-frame,
#target_col_preview .shiny-data-frame {
    display: block !important;
    text-align: left !important;
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
}

#input_col_preview .shiny-data-frame > div,
#target_col_preview .shiny-data-frame > div {
    display: block !important;
    text-align: left !important;
    margin: 0 auto 0 0 !important; /* This forces left alignment by removing auto centering */
    padding: 0 !important;
}

#input_col_preview .shiny-data-frame table,
#target_col_preview .shiny-data-frame table {
    width: 100% !important;
    table-layout: fixed !important;
    margin: 0 !important;
    margin-left: 0 !important;
    margin-right: auto !important;
    border-collapse: collapse !important;
}

/* Headers - force left alignment and dark mode support */
#input_col_preview .shiny-data-frame thead th,
#target_col_preview .shiny-data-frame thead th {
    text-align: left !important;
    padding: 8px !important;
    vertical-align: middle !important;
    font-weight: bold !important;
    color: var(--bs-body-color, #212529) !important;
    background-color: var(--bs-gray-200, #e9ecef) !important;
    border-bottom: 2px solid var(--bs-border-color, #dee2e6) !important;
}

/* First column header (Row) - centered and narrow */
#input_col_preview .shiny-data-frame thead th:first-child,
#target_col_preview .shiny-data-frame thead th:first-child {
    width: 60px !important;
    min-width: 60px !important;
    max-width: 60px !important;
    text-align: center !important;
}

/* Second column header (Sample Values) - left aligned, takes remaining space */
#input_col_preview .shiny-data-frame thead th:nth-child(2),
#target_col_preview .shiny-data-frame thead th:nth-child(2) {
    text-align: left !important;
    padding-left: 12px !important;
    width: auto !important;
}

/* Data cells - with dark mode support */
#input_col_preview .shiny-data-frame tbody td,
#target_col_preview .shiny-data-frame tbody td {
    text-align: left !important;
    padding: 8px !important;
    vertical-align: top !important;
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
    white-space: normal !important;
    color: var(--bs-body-color, #212529) !important;
    background-color: var(--bs-body-bg, white) !important;
}

/* First column data (Row numbers) - centered */
#input_col_preview .shiny-data-frame tbody td:first-child,
#target_col_preview .shiny-data-frame tbody td:first-child {
    width: 60px !important;
    text-align: center !important;
}

/* Second column data - left aligned with wrapping */
#input_col_preview .shiny-data-frame tbody td:nth-child(2),
#target_col_preview .shiny-data-frame tbody td:nth-child(2) {
    text-align: left !important;
    white-space: normal !important;
    word-break: break-word !important;
}

/* When there are 3 columns (cleaning preview) */
#input_col_preview .shiny-data-frame thead th:nth-child(3),
#target_col_preview .shiny-data-frame thead th:nth-child(3) {
    text-align: left !important;
    width: auto !important;
}

/* Matching Configuration - Polished Production Styling (theme-friendly) */
.matching-config-card {
    background-color: var(--bs-tertiary-bg, var(--bs-body-bg));
    border-radius: 12px;
    padding: 1.25rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border: 1px solid var(--bs-border-color, rgba(0,0,0,0.05));
}

.matching-config-card h5 {
    color: var(--primary-dark);
    font-weight: 600;
    letter-spacing: 0.5px;
}

.matching-config-card h6 {
    color: var(--primary-color);
    font-weight: 500;
}

.matching-config-card .border-end {
    border-color: var(--border-color) !important;
}

.matching-config-card .border-start {
    border-color: var(--border-color) !important;
}

/* Slider styling for production look - centered */
/* Ensure slider stays centered and doesn't break on resize */
.matching-config-card .d-flex.justify-content-center {
    display: flex !important;
    justify-content: center !important;
}

.matching-config-card .mx-auto {
    margin-left: auto !important;
    margin-right: auto !important;
    width: 100% !important;
}

.matching-config-card .irs-bar {
    background: var(--primary-color);
}

.matching-config-card .irs-handle {
    border: 3px solid var(--primary-color);
}

/* Centered description text */
.threshold-description {
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.5;
    color: var(--text-secondary);
}

/* Start button hover effect */
.matching-config-card .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(71, 85, 105, 0.3); }

/* Progress indicator */
.progress-container {
    background: white;
    border-radius: 12px;
    padding: 2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    margin: 1rem 0;
}

.progress {
    background-color: #e9ecef;
    border-radius: 10px;
    overflow: hidden;
}

.progress-bar {
    background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
    border-radius: 8px;
    transition: width 0.5s ease-in-out;
    font-size: 14px;
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
}

.progress-bar-striped {
    background-image: linear-gradient(45deg, rgba(255,255,255,.15) 25%, transparent 25%, transparent 50%, rgba(255,255,255,.15) 50%, rgba(255,255,255,.15) 75%, transparent 75%, transparent);
    background-size: 1rem 1rem;
}

.progress-bar-animated {
    animation: progress-bar-stripes 1s linear infinite;
}

@keyframes progress-bar-stripes {
    from { background-position: 1rem 0; }
    to { background-position: 0 0; }
}

/* Status messages */
.alert-custom {
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    border-left: 4px solid;
}

.alert-success-custom {
    background: #f0fdf4;
    border-color: var(--success-color);
    color: #166534;
}

.alert-warning-custom {
    background: #fffbeb;
    border-color: var(--warning-color);
    color: #92400e;
}

.alert-info-custom {
    background: #f0f9ff;
    border-color: var(--primary-color);
    color: var(--primary-dark);
}

/* NEW: Enhanced table with proper alignment */
.table {
    width: 100%;
    table-layout: auto;
    border-collapse: separate;
    border-spacing: 0;
    background: var(--surface);
    color: var(--text-primary);
}

.table thead {
    background: var(--surface);
    color: var(--text-primary);
    position: sticky;
    top: 0;
    z-index: 10;
    border-bottom: 2px solid var(--primary-color);
}

.table thead th {
    padding: 12px;
    font-weight: 700;
    text-align: left;
    white-space: nowrap;
    border-bottom: 2px solid var(--primary-color);
}

/* NEW: Numeric column alignment */
.table th.num,
.table td.num {
    text-align: right;
    font-variant-numeric: tabular-nums;
}

.table th.text,
.table td.text {
    text-align: left;
}

.table tbody td {
    padding: 12px;
    vertical-align: middle;
    border-bottom: 1px solid var(--border-color);
}

/* NEW: Compact density mode */
#results_container.compact .table tbody td {
    padding: 6px 12px;
    font-size: 0.875rem;
    line-height: 1.25;
}

#results_container.compact .table thead th {
    padding: 8px 12px;
}

.table-striped tbody tr:nth-child(odd) {
    background: rgba(0, 0, 0, 0.02);
}

.table tbody tr:hover {
    background: rgba(71, 85, 105, 0.05) !important;
    cursor: pointer;
}

/* Footer */
.footer {
    margin-top: 3rem;
    padding: 2rem;
    border-top: 2px solid var(--border-color);
    text-align: center;
    color: #64748b;
}

/* Animated alerts */
.alert-animated { animation: fadeIn 0.25s ease-out both; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(4px);} to { opacity: 1; transform: none; } }

/* NEW: Results container with better scrolling */
.results-container {
    height: 520px;
    overflow: auto;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 0;
    background: var(--surface);
    box-shadow: var(--shadow-sm);
    position: relative;
}

.results-container::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

.results-container::-webkit-scrollbar-track {
    background: var(--surface);
    border-radius: 10px;
}

.results-container::-webkit-scrollbar-thumb {
    background: var(--primary-color);
    border-radius: 10px;
}

.results-container::-webkit-scrollbar-thumb:hover {
    background: var(--primary-dark);
}

/* NEW: Make Shiny progress panel wider so text doesn't wrap */
.shiny-progress-panel { min-width: 520px !important; width: 520px !important; }
.shiny-progress-panel .progress { height: 0.75rem; }
.shiny-progress-panel p { white-space: normal; }

/* Loading spinner */
.spinner-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 2rem;
}

.spinner {
    width: 40px;
    height: 40px;
    border: 3px solid var(--border-color);
    border-top-color: var(--primary-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Score bar visualization */
.scorebar {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    letter-spacing: 0.5px;
    white-space: nowrap;
    color: var(--secondary-color);
}

/* NEW: Status badges */
.status-badge {
    display: inline-block;
    font-weight: 600;
    font-size: 0.75rem;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    text-transform: uppercase;
    letter-spacing: 0.025em;
}

.status-ok {
    background: rgba(16, 185, 129, 0.1);
    color: var(--success-color);
    border: 1px solid var(--success-color);
}

.status-warn {
    background: rgba(245, 158, 11, 0.1);
    color: var(--warning-color);
    border: 1px solid var(--warning-color);
}

/* Method chips */
.method-chip {
    display: inline-block;
    background: var(--secondary-color);
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 16px;
    font-size: 0.875rem;
    font-weight: 500;
    margin: 0.25rem;
}

/* NEW: Nav tabs styling */
.nav-tabs {
    border-bottom: 2px solid var(--border-color);
    margin-bottom: 1.5rem;
}

.nav-tabs .nav-link {
    color: var(--primary-color);
    border: none;
    padding: 0.75rem 1.5rem;
    font-weight: 500;
    transition: all 0.3s ease;
    position: relative;
}

.nav-tabs .nav-link:hover {
    color: var(--primary-color);
    background: transparent;
}

.nav-tabs .nav-link.active {
    color: var(--primary-dark);
    background: transparent;
    border: none;
    font-weight: 600;
}

.nav-tabs .nav-link.active::after {
    content: '';
    position: absolute;
    bottom: -2px;
    left: 0;
    right: 0;
    height: 2px;
    background: var(--primary-color);
}

/* NEW: Tooltip styles */
.tooltip {
    font-size: 0.875rem;
}

/* NEW: Mobile responsive improvements */
@media (max-width: 768px) {
    .container-fluid {
        padding: 12px;
    }
    
    .app-header {
        padding: 2rem 1rem;
        border-radius: 12px;
    }
    
    .card {
        margin-bottom: 1rem;
        border-radius: 12px;
    }
    
    .btn {
        padding: 0.625rem 1.25rem;
        font-size: 0.875rem;
    }
    .btn-group-export { justify-content: center; }
    
    .btn-group-export {
        flex-direction: column;
    }
    
    .results-container {
        height: 400px;
    }
    
    .nav-tabs .nav-link {
        padding: 0.5rem 0.75rem;
        font-size: 0.875rem;
    }
}

@media (max-width: 480px) {
    .app-header h1 {
        font-size: 1.75rem;
    }
    
    .app-header p {
        font-size: 0.875rem;
    }
    
    .table {
        font-size: 0.75rem;
    }
    
    .table thead th,
    .table tbody td {
        padding: 8px 6px;
    }
}

/* Simple transitions */
.fade-in {
    animation: fadeIn 0.2s ease;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* NEW: File input styling */
.file-upload-area {
    border: 2px dashed var(--border-color);
    border-radius: 12px;
    padding: 2rem;
    background: var(--surface);
    text-align: center;
    transition: all 0.3s ease;
    cursor: pointer;
}

.file-upload-area:hover {
    border-color: var(--primary-color);
    background: var(--background);
    box-shadow: var(--shadow-sm);
}

/* Alert animations */
.alert-animated {
    animation: fadeIn 0.2s ease;
}

/* Fix Shiny's file input progress bar - center text vertically */
.shiny-input-container .progress {
    min-height: 1.5rem !important;
    height: 1.5rem !important;
    display: flex !important;
    align-items: center !important;
}

.shiny-input-container .progress-bar {
    min-height: 1.5rem !important;
    height: 1.5rem !important;
    line-height: 1 !important;  /* Reset line-height */
    font-size: 0.875rem;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 0.5rem;
}

/* Make the 'Loaded X rows' status alerts thinner */
#input_file_status .alert,
#target_file_status .alert {
    padding: 0.4rem 0.75rem !important;
    margin-bottom: 0.25rem;
    font-size: 0.875rem;
}

#input_file_status p,
#target_file_status p {
    margin: 0;
    line-height: 1.2;
}
"""

# Minimal custom CSS to preserve app behaviors while letting themes show
custom_css_min = """
/* Fix Shiny's file input progress bar - thinner with centered text */
.sidebar { min-height: 100vh; }

.page-sidebar .sidebar { min-height: 100vh; }

/* Small, theme-friendly footer shown on all pages */
.footer {
    margin-top: 1rem !important;
    padding: 0.75rem 1rem !important;
    border-top: 1px solid var(--bs-border-color, #dee2e6) !important;
    text-align: center !important;
    color: var(--bs-secondary-color, inherit) !important;
    font-size: 0.9rem !important;
}

.shiny-input-container .progress {
    min-height: 1.4rem !important;
    height: 1.4rem !important;
    margin-bottom: 0 !important;  /* Remove bottom margin */
}

/* Let Shiny's default animation work - only adjust height and text position */
.shiny-input-container .progress-bar {
    min-height: 1.4rem !important;
    height: 1.4rem !important;
    line-height: 1.4rem !important;
    font-size: 0.8rem !important;
    padding-top: 0.1rem !important;  /* Small padding to center text */
    /* Allow Shiny's default transition animation */
    transition: width 0.6s ease !important;
}

/* Make the 'Loaded X rows' status alerts much thinner and even closer to upload bar */
#input_file_status .alert,
#target_file_status .alert {
    padding: 0.25rem 0.5rem !important;
    margin-bottom: 0.5rem !important;
    margin-top: -0.75rem !important;  /* Even more negative margin to bring closer */
    font-size: 0.85rem !important;
    line-height: 1.1 !important;
    min-height: auto !important;
    position: relative !important;
    top: -0.25rem !important;  /* Additional upward shift */
}

/* Also adjust the container divs to reduce spacing */
#input_status,
#target_status {
    margin-top: -0.25rem !important;
    margin-bottom: 0 !important;
    padding-top: 0 !important;
}

#input_file_status p,
#target_file_status p {
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.1 !important;
}

/* Results container: bounded height and scrolling */
.results-container {
    height: 520px;
    overflow: auto;
}

/* Compact density mode for results table */
#results_container.compact table tbody td {
    padding: 6px 12px;
    font-size: 0.875rem;
    line-height: 1.25;
}
#results_container.compact table thead th {
    padding: 8px 12px;
}

/* Numeric/text alignment used by JS alignment helper */
.table th.num, .table td.num { text-align: right; font-variant-numeric: tabular-nums; }
.table th.text, .table td.text { text-align: left; }

/* Score bar visualization */
.scorebar {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    white-space: nowrap;
    letter-spacing: 0.5px;
}

/* Simple spinner for long-running tasks */
.spinner-container { display: flex; justify-content: center; align-items: center; padding: 2rem; }
.spinner { width: 40px; height: 40px; border: 3px solid rgba(0,0,0,0.1); border-top-color: currentColor; border-radius: 50%; animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

/* Export button group layout */
.btn-group-export { display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; }

/* Optional small fade-in animation class used by notifications */
.alert-animated { animation: fadeIn 0.25s ease-out both; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(4px);} to { opacity: 1; transform: none; } }

/* Shiny progress panel: polished and theme-adaptive */
.shiny-progress-panel {
    width: 420px !important;
    min-width: 360px !important;
    max-width: 90vw;
    padding: 1.25rem;
    border-radius: 0.5rem;
    background-color: var(--bs-body-bg, white) !important;
    border: 1px solid var(--bs-border-color, rgba(0,0,0,0.125)) !important;
    box-shadow: 0 0.5rem 1rem rgba(0,0,0,0.15), 0 0.125rem 0.25rem rgba(0,0,0,0.075);
    animation: slide-in-bottom 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94) both;
    /* Position at bottom right with proper spacing */
    position: fixed !important;
    right: 20px !important;
    bottom: 20px !important;
    left: auto !important;
    top: auto !important;
    transform: none !important;
    z-index: 9999 !important;
}
.shiny-progress-panel .progress-text,
.shiny-progress-panel p { 
    color: var(--bs-body-color, #212529) !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    margin-bottom: 0.75rem !important;
    line-height: 1.4 !important;
    display: block !important;
    text-align: left !important;
}
.shiny-progress-panel .progress { 
    height: 1.25rem !important;
    background-color: var(--bs-gray-200, #e9ecef) !important;
    border-radius: 0.375rem !important;
    overflow: hidden !important;
    margin-bottom: 0.5rem !important;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.075) !important;
}
.shiny-progress-panel .progress-bar { 
    background: linear-gradient(90deg, var(--bs-primary, #0d6efd), var(--bs-info, #0dcaf0)) !important;
    transition: width 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    color: white !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    position: relative !important;
    overflow: hidden !important;
}
.shiny-progress-panel .progress-bar::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    bottom: 0;
    right: 0;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(255, 255, 255, 0.2),
        transparent
    );
    animation: shimmer 2s infinite;
}
@keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}
.shiny-progress-panel .progress-bar-animated { 
    background-image: linear-gradient(
        45deg,
        rgba(255,255,255,.15) 25%,
        transparent 25%,
        transparent 50%,
        rgba(255,255,255,.15) 50%,
        rgba(255,255,255,.15) 75%,
        transparent 75%,
        transparent
    ) !important;
    background-size: 1rem 1rem !important;
    animation: progress-bar-stripes 1s linear infinite !important;
}
@keyframes slide-in-bottom {
    0% {
        transform: translateY(50px);
        opacity: 0;
    }
    100% {
        transform: translateY(0);
        opacity: 1;
    }
}
@keyframes progress-bar-stripes { 
    from { background-position: 1rem 0; } 
    to { background-position: 0 0; } 
}

/* Enhanced navbar tab styling */
.navbar-nav .nav-item .nav-link {
    border-radius: 0.375rem;
    padding: 0.5rem 1rem !important;
    margin: 0 0.25rem;
    transition: all 0.2s ease;
    position: relative;
    color: var(--bs-nav-link-color, #495057) !important;
}

.navbar-nav .nav-item .nav-link.active {
    background-color: var(--bs-primary, #0d6efd) !important;
    color: white !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.navbar-nav .nav-item .nav-link:hover:not(.active) {
    background-color: var(--bs-gray-200, #e9ecef);
    color: var(--bs-body-color, #212529) !important;
}

/* Remove default underline */
.navbar-nav .nav-item .nav-link.active::after {
    display: none !important;
}

/* Matching configuration panel - theme friendly */
.config-card {
    background-color: var(--bs-tertiary-bg, var(--bs-body-bg)) !important;
    border: 1px solid var(--bs-border-color, #dee2e6) !important;
    border-radius: .5rem !important;
    padding: 1rem !important;
}
/* threshold-badge removed (using slider only) */
.config-row .btn { width: 100%; }

/* Step 1 preview tables: keep left-aligned and full width at all sizes */
#input_col_preview, #target_col_preview {
    text-align: left !important;
    display: block !important;
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
}
#input_preview_wrap, #target_preview_wrap {
    display: flex !important;
    width: 100% !important;
}
#input_preview_wrap > *, #target_preview_wrap > * {
    flex: 1 1 auto !important;
    width: 100% !important;
}
#input_col_preview .shiny-data-frame, #target_col_preview .shiny-data-frame {
    display: block !important;
    text-align: left !important;
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
}
#input_col_preview .gridjs-container, #target_col_preview .gridjs-container,
#input_col_preview .gridjs-wrapper, #target_col_preview .gridjs-wrapper {
    width: 100% !important;
    max-width: none !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
}
/* Improve text density so more characters fit on two lines */
#input_col_preview .gridjs-table td.gridjs-td,
#target_col_preview .gridjs-table td.gridjs-td {
    padding-left: 6px !important;
    padding-right: 6px !important;
    font-size: 0.95rem !important;
    line-height: 1.2 !important;
    white-space: normal !important;
    word-break: break-word !important;
    overflow-wrap: anywhere !important;
    hyphens: auto !important;
}
#input_col_preview .gridjs-table th.gridjs-th,
#target_col_preview .gridjs-table th.gridjs-th {
    padding-left: 8px !important;
    padding-right: 8px !important;
}
#input_col_preview .shiny-data-frame > div, #target_col_preview .shiny-data-frame > div {
    display: block !important;
    text-align: left !important;
    margin: 0 auto 0 0 !important; /* prevent auto-centering */
    padding: 0 !important;
    width: 100% !important;
    max-width: none !important;
}
#input_col_preview .shiny-data-frame table, #target_col_preview .shiny-data-frame table,
#input_col_preview .gridjs-table, #target_col_preview .gridjs-table {
    width: 100% !important;
    /* Allow content to determine width so text columns expand and 'Row' stays narrow */
    table-layout: auto !important;
    margin: 0 !important;
    margin-left: 0 !important;
    margin-right: auto !important;
    border-collapse: collapse !important;
}
#input_preview_wrap .table-responsive, #target_preview_wrap .table-responsive {
    width: 100% !important;
    margin: 0 !important;
}
/* Make any Shiny fill wrappers align to the left and fill width */
#input_col_preview .html-fill-container, #target_col_preview .html-fill-container,
#input_col_preview .html-fill-item, #target_col_preview .html-fill-item {
    display: block !important;
    justify-content: flex-start !important;
    align-items: stretch !important;
    width: 100% !important;
    max-width: none !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
}
/* Catch-all for inline auto-centering styles */
#input_col_preview div[style*="margin: 0 auto"],
#target_col_preview div[style*="margin: 0 auto"],
#input_col_preview div[style*="margin-left: auto"],
#input_col_preview div[style*="margin-right: auto"],
#target_col_preview div[style*="margin-left: auto"],
#target_col_preview div[style*="margin-right: auto"] {
    margin-left: 0 !important;
    margin-right: 0 !important;
    width: 100% !important;
    max-width: none !important;
}
/* Shiny fill layout elements sometimes center children with auto margins */
#input_col_preview .html-fill-item, #target_col_preview .html-fill-item,
#input_col_preview .html-fill-container, #target_col_preview .html-fill-container {
    margin-left: 0 !important;
    margin-right: 0 !important;
    width: 100% !important;
    max-width: none !important;
}
/* In case an inner wrapper uses inline style widths, let it stretch */
#input_col_preview div[style*="margin-left: auto"],
#input_col_preview div[style*="margin-right: auto"],
#target_col_preview div[style*="margin-left: auto"],
#target_col_preview div[style*="margin-right: auto"] {
    margin-left: 0 !important;
    margin-right: 0 !important;
    width: 100% !important;
}
/* Header/data alignment for consistency */
#input_col_preview .shiny-data-frame thead th,
#target_col_preview .shiny-data-frame thead th { text-align: left !important; }
#input_col_preview .shiny-data-frame thead th:first-child,
#target_col_preview .shiny-data-frame thead th:first-child { width: 36px !important; text-align: center !important; }
#input_col_preview .shiny-data-frame tbody td,
#target_col_preview .shiny-data-frame tbody td { text-align: left !important; vertical-align: top !important; }
#input_col_preview .shiny-data-frame tbody td:first-child,
#target_col_preview .shiny-data-frame tbody td:first-child { width: 36px !important; text-align: center !important; white-space: nowrap !important; }

/* Also constrain first column via colgroup to override library sizing */
#input_col_preview .shiny-data-frame colgroup col:first-child,
#target_col_preview .shiny-data-frame colgroup col:first-child {
    width: 36px !important;
    min-width: 36px !important;
    max-width: 36px !important;
}
/* Tighten padding on the small first column */
#input_col_preview .shiny-data-frame thead th:first-child,
#input_col_preview .shiny-data-frame tbody td:first-child,
#target_col_preview .shiny-data-frame thead th:first-child,
#target_col_preview .shiny-data-frame tbody td:first-child {
    padding-left: 6px !important;
    padding-right: 6px !important;
}

/* Grid.js (Shiny DataGrid) often enforces a min column width (~120px).
   Explicitly override only for the first column within these two previews. */
#input_col_preview .gridjs-table thead th:first-child,
#input_col_preview .gridjs-table tbody td:first-child,
#input_col_preview .gridjs-header .gridjs-th:first-child,
#input_col_preview .gridjs-body .gridjs-td:first-child,
#target_col_preview .gridjs-table thead th:first-child,
#target_col_preview .gridjs-table tbody td:first-child,
#target_col_preview .gridjs-header .gridjs-th:first-child,
#target_col_preview .gridjs-body .gridjs-td:first-child {
    width: 36px !important;
    min-width: 36px !important;
    max-width: 40px !important;
    text-align: center !important;
    white-space: nowrap !important;
}

/* Make sure the table can use the freed space for text columns */
#input_col_preview .gridjs-table,
#target_col_preview .gridjs-table { width: 100% !important; }

/* Center overlay progress (theme-friendly) */
.center-progress-backdrop { position: fixed; inset: 0; background: rgba(0,0,0,0.25); display: flex; align-items: center; justify-content: center; z-index: 2000; }
.center-progress-panel { width: 720px; max-width: 90vw; }
/* App title size */
.app-title { font-size: 2rem; font-weight: 700; margin: 0; }

/* Highlight NO MATCH rows with light red background */
.no-match-row {
    background-color: rgba(220, 53, 69, 0.08) !important;
}
.no-match-row:hover {
    background-color: rgba(220, 53, 69, 0.15) !important;
}
"""

# ============================================================================
# EMBEDDING CONFIGURATION
# ============================================================================

# Model and API settings
DEEPINFRA_MODEL = "thenlper/gte-large"
# Concurrency settings
MAX_CONCURRENCY = int(os.environ.get("EMBEDDING_CONCURRENCY", "100"))
EMBED_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "200"))
USE_PRIORITY_TIER = os.environ.get("DEEPINFRA_PRIORITY", "false").lower() in {"1", "true", "yes", "on"}
USE_ASYNC = os.environ.get("EMBEDDING_ASYNC", "true").lower() in {"1", "true", "yes", "on"}

# Fallback behavior
API_EMBED_TIMEOUT_SECS = int(os.environ.get("API_EMBED_TIMEOUT_SECS", "45"))  # overall call timeout
API_MAX_FAILURES = int(os.environ.get("API_EMBED_MAX_FAILURES", "3"))         # consecutive failures before CPU fallback
MODEL_FALLBACK_MODE = os.environ.get("MODEL_FALLBACK_MODE", "auto").lower()
# Values: "auto" (try API then fallback), "api" (force API only), "local" (force CPU), "off" (no fallback)

# Runtime state
FALLBACK_ACTIVE: bool = False
_API_FAILURES: int = 0
_LOCAL_ST_MODEL: Optional["SentenceTransformer"] = None  # lazy-loaded cache

def get_api_key():
    """Get API key from environment variable or HuggingFace secret"""
    # Try HuggingFace secret first
    api_key = os.environ.get("DEEPINFRA_API_KEY")
    if not api_key:
        # Try standard environment variable
        api_key = os.environ.get("DEEPINFRA_TOKEN")
    return api_key

# Client caching for connection pooling
_CLIENT_CACHE: Dict[str, OpenAI] = {}
_ASYNC_CLIENT_CACHE: Dict[str, AsyncOpenAI] = {}

def get_openai_client(api_key: str) -> OpenAI:
    """Create or retrieve cached OpenAI client configured for DeepInfra"""
    if api_key in _CLIENT_CACHE:
        return _CLIENT_CACHE[api_key]
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepinfra.com/v1/openai"
    )
    _CLIENT_CACHE[api_key] = client
    return client

def get_async_openai_client(api_key: str) -> AsyncOpenAI:
    """Create or retrieve cached AsyncOpenAI client configured for DeepInfra"""
    if api_key in _ASYNC_CLIENT_CACHE:
        return _ASYNC_CLIENT_CACHE[api_key]
    # Configure HTTP client for higher concurrency
    limits = httpx.Limits(
        max_connections=max(10, MAX_CONCURRENCY),
        max_keepalive_connections=max(10, MAX_CONCURRENCY),
    )
    timeout = httpx.Timeout(60.0, connect=20.0, read=60.0, write=60.0)
    # Enable HTTP/2 only if the 'h2' package is available
    http_client = httpx.AsyncClient(limits=limits, timeout=timeout, http2=HTTP2_AVAILABLE)
    if not HTTP2_AVAILABLE:
        print("[async] HTTP/2 not available (h2 not installed). Falling back to HTTP/1.1")
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepinfra.com/v1/openai",
        http_client=http_client,
    )
    _ASYNC_CLIENT_CACHE[api_key] = client
    return client

def compute_embeddings_deepinfra(texts: List[str], api_key: str) -> np.ndarray:
    """Compute embeddings using DeepInfra API via OpenAI client"""
    client = get_openai_client(api_key)

    try:
        extra_body = {"normalize": True}
        if USE_PRIORITY_TIER:
            extra_body["service_tier"] = "priority"

        # Create embeddings using OpenAI client
        response = client.embeddings.create(
            model=DEEPINFRA_MODEL,
            input=texts,
            encoding_format="float",
            extra_body=extra_body,
        ),

        # Extract embeddings from response (already normalized if normalize=True)
        embeddings = np.array([data.embedding for data in response.data], dtype=np.float32)
        return embeddings

    except Exception as e:
        raise Exception(f"DeepInfra API error: {str(e)}")

async def compute_embeddings_deepinfra_async(texts: List[str], api_key: str) -> np.ndarray:
    """Async embeddings using DeepInfra via AsyncOpenAI client"""
    client = get_async_openai_client(api_key)
    try:
        extra_body = {"normalize": True}
        if USE_PRIORITY_TIER:
            extra_body["service_tier"] = "priority"
        # Retry loop to avoid transient throttling
        last_err = None
        for attempt in range(5):
            try:
                t0 = time.perf_counter()
                response = await client.embeddings.create(
                    model=DEEPINFRA_MODEL,
                    input=texts,
                    encoding_format="float",
                    extra_body=extra_body,
                )
                dt = time.perf_counter() - t0
                embeddings = np.array([data.embedding for data in response.data], dtype=np.float32)
                print(f"[async] embeddings.create batch_size={len(texts)} took {dt:.2f}s")
                return embeddings
            except Exception as e:
                last_err = e
                await asyncio.sleep(min(4.0, 0.25 * (2 ** attempt)))
        raise Exception(f"DeepInfra API error after retries: {str(last_err)}")
    except Exception as e:
        raise Exception(f"DeepInfra API error: {str(e)}")

def _chunk_indices(total: int, chunk_size: int) -> List[Tuple[int, int]]:
    return [(i, min(i + chunk_size, total)) for i in range(0, total, chunk_size)]

def _embed_batch_slice(args: Tuple[int, int, List[str], str]) -> Tuple[int, np.ndarray]:
    start, end, texts, api_key = args
    batch_vecs = compute_embeddings_deepinfra(texts[start:end], api_key)
    return start, batch_vecs

def compute_embeddings_parallel(
    texts: List[str],
    api_key: str,
    batch_size: int = EMBED_BATCH_SIZE,
    max_concurrency: int = MAX_CONCURRENCY,
    progress_callback=None,
) -> np.ndarray:
    """Concurrent embedding across batches while preserving order."""
    n = len(texts)
    if n == 0:
        return np.empty((0, 0), dtype=np.float32)

    slices = _chunk_indices(n, batch_size)
    results: Dict[int, np.ndarray] = {}

    total_batches = len(slices)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_concurrency) as ex:
        futures = [
            ex.submit(_embed_batch_slice, (start, end, texts, api_key))
            for (start, end) in slices
        ]
        for fut in as_completed(futures):
            start, vecs = fut.result()
            results[start] = vecs
            completed += 1
            if progress_callback:
                pct = int((completed / total_batches) * 100)
                progress_callback(f"Embedding batches: {pct}% ({completed}/{total_batches})")

    # Assemble in order
    ordered_starts = sorted(results.keys())
    assembled = np.vstack([results[s] for s in ordered_starts])
    return assembled

async def compute_embeddings_parallel_async(
    texts: List[str],
    api_key: str,
    batch_size: int = EMBED_BATCH_SIZE,
    max_concurrency: int = MAX_CONCURRENCY,
    progress_callback=None,
) -> np.ndarray:
    """Async concurrent embedding across batches while preserving order."""
    n = len(texts)
    if n == 0:
        return np.empty((0, 0), dtype=np.float32)
    slices = _chunk_indices(n, batch_size)
    results: Dict[int, np.ndarray] = {}
    sem = asyncio.Semaphore(max_concurrency)
    total_batches = len(slices)
    completed = 0

    async def worker(start: int, end: int):
        nonlocal completed
        print(f"[async] launch target slice {start}:{end}")
        async with sem:
            vecs = await compute_embeddings_deepinfra_async(texts[start:end], api_key)
        print(f"[async] done target slice {start}:{end}")
        results[start] = vecs
        completed += 1
        if progress_callback:
            pct = int((completed / total_batches) * 100)
            progress_callback(f"Embedding batches: {pct}% ({completed}/{total_batches})")

    await asyncio.gather(*(worker(start, end) for (start, end) in slices))
    ordered_starts = sorted(results.keys())
    return np.vstack([results[s] for s in ordered_starts])

# Local CPU Embedding Backend (async-compatible)
async def _load_local_model() -> "SentenceTransformer":
    global _LOCAL_ST_MODEL
    if _LOCAL_ST_MODEL is not None:
        return _LOCAL_ST_MODEL
    
    # Show notification that model is loading (may need to download)
    try:
        from shiny import ui
        ui.notification_show(
            "Loading local embedding model (thenlper/gte-large). "
            "First-time loading may take a few minutes to download the model (~670MB).",
            type="info",
            duration=None,  # Keep showing until we dismiss it
            id="model_loading"
        )
    except:
        pass
    
    # Lazy import to avoid heavy import if API is healthy
    from sentence_transformers import SentenceTransformer
    # Load CPU model (default behavior). This may take time on first run (download + init).
    model = await asyncio.to_thread(SentenceTransformer, "thenlper/gte-large")
    _LOCAL_ST_MODEL = model
    
    # Dismiss loading notification
    try:
        from shiny import ui
        ui.notification_remove("model_loading")
        ui.notification_show(
            "Local embedding model loaded successfully.",
            type="success",
            duration=3
        )
    except:
        pass
    
    return model

async def compute_embeddings_local_async(texts: List[str]) -> np.ndarray:
    # Minimal cleaning consistent with your embedding path
    texts = clean_text_for_embedding(texts)
    if len(texts) == 0:
        return np.empty((0, 0), dtype=np.float32)

    model = await _load_local_model()
    # Batch via your existing chunking to keep memory bounded
    slices = _chunk_indices(len(texts), EMBED_BATCH_SIZE)
    results: Dict[int, np.ndarray] = {}
    completed = 0
    total = len(slices)

    async def work(start: int, end: int):
        # Run CPU-bound encode in a thread to keep event loop responsive
        vecs = await asyncio.to_thread(model.encode, texts[start:end], normalize_embeddings=True)
        # vecs is a numpy array
        results[start] = vecs.astype(np.float32, copy=False)

    await asyncio.gather(*(work(s, e) for (s, e) in slices))
    # Assemble in order
    ordered = [results[s] for s in sorted(results.keys())]
    return np.vstack(ordered) if ordered else np.empty((0, 0), dtype=np.float32)

# Resilient Wrapper (API first, CPU fallback)
async def _try_api_embeddings(texts: List[str], api_key: str, progress_callback=None) -> np.ndarray:
    # Wrap your existing async API call with a timeout
    coro = compute_embeddings_deepinfra_async(texts, api_key)
    return await asyncio.wait_for(coro, timeout=API_EMBED_TIMEOUT_SECS)

async def compute_embeddings_resilient_async(
    texts: List[str],
    api_key: str,
    progress_callback=None,
) -> np.ndarray:
    global FALLBACK_ACTIVE, _API_FAILURES

    mode = MODEL_FALLBACK_MODE  # "auto" | "api" | "local" | "off"

    # Force-local mode
    if mode == "local":
        FALLBACK_ACTIVE = True
        if progress_callback:
            progress_callback("Local CPU embeddings (forced).")
        return await compute_embeddings_local_async(texts)

    # Force-API mode
    if mode == "api":
        FALLBACK_ACTIVE = False
        if progress_callback:
            progress_callback("Using API embeddings (forced).")
        return await _try_api_embeddings(texts, api_key, progress_callback)

    # Fallback disabled entirely
    if mode == "off":
        FALLBACK_ACTIVE = False
        return await _try_api_embeddings(texts, api_key, progress_callback)

    # Auto mode: try API, fallback on failure or repeated errors
    if FALLBACK_ACTIVE:
        # Circuit open: stay on local until next run
        if progress_callback:
            progress_callback("Local CPU embeddings (fallback active).")
        return await compute_embeddings_local_async(texts)

    try:
        vecs = await _try_api_embeddings(texts, api_key, progress_callback)
        # On success, reset failure counter
        _API_FAILURES = 0
        FALLBACK_ACTIVE = False
        return vecs
    except Exception as e:
        _API_FAILURES += 1
        if _API_FAILURES >= API_MAX_FAILURES:
            FALLBACK_ACTIVE = True
            # Show immediate notification when switching to CPU fallback
            try:
                from shiny import ui
                ui.notification_show(
                    f"API failed after {_API_FAILURES} attempts. Switching to LOCAL CPU processing. "
                    f"This will be much slower. Processing {len(texts)} items may take several minutes.",
                    type="warning",
                    duration=10,
                )
            except:
                pass  # ui might not be available in all contexts
            # Loggable note
            if progress_callback:
                progress_callback("API unavailable. Falling back to local CPU.")
            return await compute_embeddings_local_async(texts)
        # Re-raise before we hit threshold so upstream can decide (e.g., show an error or retry)
        raise

def clean_text_simple(text_list: List[str]) -> List[str]:
    """Clean text by removing punctuation and extra spaces"""
    cleaned = []
    for text in text_list:
        text = str(text).strip()
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'[^\w\s,.-]', '', text)  # Keep basic punctuation
        cleaned.append(text.lower())
    return cleaned

def clean_text_for_embedding(text_list: List[str]) -> List[str]:
    """Minimal cleaning for embedding models"""
    cleaned = []
    for text in text_list:
        text = str(text).strip()
        text = re.sub(r'\s+', ' ', text)
        cleaned.append(text)
    return cleaned

def run_fuzzy_match(input_list: List[str], target_list: List[str], clean: bool = True) -> Dict:
    """Run fuzzy string matching"""
    if clean:
        input_list = clean_text_simple(input_list)
        target_list = clean_text_simple(target_list)
    
    matches = []
    scores = []
    
    for input_desc in input_list:
        best_match, score, _ = process.extractOne(
            input_desc, 
            target_list, 
            scorer=fuzz.ratio
        )
        matches.append(best_match)
        scores.append(score / 100.0)  # Normalize to 0-1
    
    return {"match": matches, "score": scores}

def run_tfidf_match(input_list: List[str], target_list: List[str], clean: bool = True) -> Dict:
    """Run TF-IDF matching with cosine similarity"""
    if clean:
        input_list = clean_text_simple(input_list)
        target_list = clean_text_simple(target_list)
    
    combined = input_list + target_list
    
    vectorizer = TfidfVectorizer()
    vectorizer.fit(combined)
    
    tfidf_input = vectorizer.transform(input_list)
    tfidf_target = vectorizer.transform(target_list)
    
    similarity_matrix = cosine_similarity(tfidf_input, tfidf_target)
    
    matches = []
    scores = []
    
    for row in similarity_matrix:
        best_idx = np.argmax(row)
        best_score = row[best_idx]
        best_match = target_list[best_idx]
        
        matches.append(best_match)
        scores.append(float(best_score))
    
    return {"match": matches, "score": scores}

def run_embed_match(
    input_list: List[str],
    target_list: List[str],
    api_key: str,
    batch_size: int = EMBED_BATCH_SIZE,
    progress_callback=None,
    max_concurrency: int = MAX_CONCURRENCY,
    clean_input: bool = False,
    clean_target: bool = False,
) -> Dict:
    """Run semantic embedding matching using DeepInfra API with concurrent batching.

    Respects DeepInfra's 1024 max batch size and uses up to `max_concurrency`
    concurrent requests to reduce wall-clock time. Results are reassembled in
    the original order.
    """
    # Apply cleaning based on user selection
    input_list_clean = clean_text_for_embedding(input_list) if clean_input else input_list
    target_list_clean = clean_text_for_embedding(target_list) if clean_target else target_list

    total_inputs = len(input_list_clean)
    total_targets = len(target_list_clean)

    # 1) Compute target embeddings concurrently (once per run)
    if progress_callback:
        progress_callback("Computing target embeddings (concurrent)...")
    target_embeddings = compute_embeddings_parallel(
        target_list_clean,
        api_key,
        batch_size=min(batch_size, 1024),
        max_concurrency=max_concurrency,
        progress_callback=progress_callback,
    )

    # 2) Prepare concurrent input embedding + local similarity
    if progress_callback:
        progress_callback("Computing input embeddings (concurrent)...")

    input_slices = _chunk_indices(total_inputs, min(batch_size, 1024))
    results_match: Dict[int, List[str]] = {}
    results_score: Dict[int, List[float]] = {}

    def _process_input_slice(args: Tuple[int, int]) -> Tuple[int, List[str], List[float]]:
        start, end = args
        emb = compute_embeddings_deepinfra(input_list_clean[start:end], api_key)
        sim = cosine_similarity(emb, target_embeddings)
        batch_matches: List[str] = []
        batch_scores: List[float] = []
        for row in sim:
            idx = int(np.argmax(row))
            batch_matches.append(target_list[idx])
            batch_scores.append(float(row[idx]))
        return start, batch_matches, batch_scores

    total_batches = len(input_slices)
    completed = 0
    with ThreadPoolExecutor(max_workers=max_concurrency) as ex:
        futures = [ex.submit(_process_input_slice, sl) for sl in input_slices]
        for fut in as_completed(futures):
            start, m, s = fut.result()
            results_match[start] = m
            results_score[start] = s
            completed += 1
            if progress_callback:
                pct = int((completed / total_batches) * 100)
                progress_callback(f"Matching: {pct}% ({completed}/{total_batches})")

    # 3) Assemble results in order
    matches: List[str] = []
    scores: List[float] = []
    for start in sorted(results_match.keys()):
        matches.extend(results_match[start])
        scores.extend(results_score[start])

    if progress_callback:
        progress_callback("Finalizing results...")

    return {"match": matches, "score": scores}

async def run_embed_match_async(
    input_list: List[str],
    target_list: List[str],
    api_key: str,
    batch_size: int = EMBED_BATCH_SIZE,
    progress_callback=None,
    max_concurrency: int = MAX_CONCURRENCY,
    clean_input: bool = False,
    clean_target: bool = False,
) -> Dict:
    """Async version using AsyncOpenAI and asyncio concurrency."""
    # Apply cleaning based on user selection
    input_list_clean = clean_text_for_embedding(input_list) if clean_input else input_list
    target_list_clean = clean_text_for_embedding(target_list) if clean_target else target_list

    # 1) Targets once
    if progress_callback:
        progress_callback("Computing target embeddings (async concurrent)...")
    target_embeddings = await compute_embeddings_resilient_async(
        target_list_clean,
        api_key,
        progress_callback=progress_callback,
    )

    # 2) Inputs concurrent and local similarity
    if progress_callback:
        progress_callback("Computing input embeddings (async concurrent)...")

    input_slices = _chunk_indices(len(input_list_clean), min(batch_size, 1024))
    results_match: Dict[int, List[str]] = {}
    results_score: Dict[int, List[float]] = {}
    sem = asyncio.Semaphore(max_concurrency)
    completed = 0
    total_batches = len(input_slices)

    async def worker(start: int, end: int):
        nonlocal completed
        print(f"[async] launch input slice {start}:{end}")
        async with sem:
            emb = await compute_embeddings_resilient_async(input_list_clean[start:end], api_key, progress_callback)
        print(f"[async] done input slice {start}:{end}")
        sim = cosine_similarity(emb, target_embeddings)
        batch_matches: List[str] = []
        batch_scores: List[float] = []
        for row in sim:
            idx = int(np.argmax(row))
            batch_matches.append(target_list[idx])
            batch_scores.append(float(row[idx]))
        results_match[start] = batch_matches
        results_score[start] = batch_scores
        completed += 1
        if progress_callback:
            pct = int((completed / total_batches) * 100)
            progress_callback(f"Matching: {pct}% ({completed}/{total_batches})")

    await asyncio.gather(*(worker(s, e) for (s, e) in input_slices))

    matches: List[str] = []
    scores: List[float] = []
    for start in sorted(results_match.keys()):
        matches.extend(results_match[start])
        scores.extend(results_score[start])
    if progress_callback:
        progress_callback("Finalizing results...")
    return {"match": matches, "score": scores}

def get_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get sample datasets for demonstration"""
    # Sample input data
    input_data = pd.DataFrame({
        "id": range(1, 26),
        "description": [
            "apple juice", "chicken breast grilled", "whole milk",
            "orange juice fresh", "bread whole wheat", "cheddar cheese", "scrambled eggs",
            "pasta with tomato sauce", "beef steak medium rare", "yogurt plain",
            "brown rice cooked", "salmon fillet baked",
            "vegetable soup", "fruit salad mixed", "pizza slice pepperoni",
            "ice cream vanilla", "coffee with cream",
            "energy drink", "protein bar chocolate", "trail mix nuts", "smoothie berry",
            "xyz123 test item", "random text here", "unknown food item 999", "synthetic compound ABC"
        ]
    })
    
    # Sample target data
    target_data = pd.DataFrame({
        "code": [f"A{i:03d}" for i in range(1, 26)],
        "reference": [
            "Apple juice, unsweetened, bottled, without added ascorbic acid",
            "Chicken, broilers or fryers, breast, meat only, cooked, grilled",
            "Milk, whole, 3.25% milkfat, with added vitamin D",
            "Orange juice, raw, includes from concentrate, fortified with calcium",
            "Bread, whole-wheat, commercially prepared",
            "Cheese, cheddar, sharp, sliced",
            "Egg, whole, cooked, scrambled",
            "Pasta with tomato-based sauce",
            "Beef, short loin, t-bone steak, separable lean and fat, trimmed to 1/8\" fat, all grades, cooked, grilled",
            "Yogurt, plain, whole milk",
            "Rice, brown, medium-grain, cooked",
            "Fish, salmon, Atlantic, farmed, cooked, dry heat",
            "Soup, vegetable with beef broth, canned, prepared with equal volume water",
            "Fruit salad, (pineapple and papaya and banana and guava), tropical, canned, heavy syrup, solids and liquids",
            "Pizza, meat and vegetable topping, regular crust, frozen, cooked",
            "Ice creams, vanilla",
            "Coffee, brewed from grounds, prepared with tap water, decaffeinated",
            "Beverages, Energy drink, RED BULL",
            "Snacks, granola bar, chocolate chip",
            "Snacks, trail mix, regular, unsalted",
            "Beverages, Smoothie, strawberry",
            "Water, tap, municipal",
            "Crackers, standard snack-type, regular",
            "Cookies, chocolate chip, commercially prepared, regular",
            "Candies, milk chocolate"
        ]
    })
    
    return input_data, target_data

# Create Shiny app with modern theme
''' LEGACY LAYOUT (disabled)
app_ui = ui.page_sidebar(
    # Sidebar must be the first positional argument
    ui.sidebar(
        ui.h5("Quick Start"),
        ui.input_action_button(
            "load_sample",
            "Load Sample Dataset",
            class_="btn btn-success w-100"
        ),
        ui.div(id="sample_status", class_="mt-2"),
        ui.hr(),
        ui.h5("Upload Data"),
        ui.input_file("input_file", "Input CSV", accept=[".csv"], multiple=False),
        ui.div(id="input_status", class_="mt-2"),
        ui.input_file("target_file", "Target CSV", accept=[".csv"], multiple=False),
        ui.div(id="target_status", class_="mt-2"),
        ui.hr(),
        ui.output_ui("sidebar_results_summary_block"),
        open="open",
    ),
    # Then page contents (positional)
    ui.tags.head(
        ui.tags.link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"),
        ui.tags.link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css"),
        ui.tags.link(rel="stylesheet", href="https://unpkg.com/tabulator-tables@5.5.2/dist/css/tabulator.min.css"),
        ui.tags.style(custom_css_min),
        ui.tags.script(src="https://unpkg.com/tabulator-tables@5.5.2/dist/js/tabulator.min.js"),
        # JavaScript for tooltips and table features
        ui.tags.script("""
            // Initialize tooltips
            document.addEventListener('DOMContentLoaded', () => {
                const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
                tooltipTriggerList.map(function (tooltipTriggerEl) {
                    return new bootstrap.Tooltip(tooltipTriggerEl);
                });
            });
            
            // Toggle compact mode for results table
            document.addEventListener('change', function(e){
                if(e.target && e.target.id === 'compact_density'){
                    const c = document.getElementById('results_container');
                    if(c){ e.target.checked ? c.classList.add('compact') : c.classList.remove('compact'); }
                }
            });

            // NEW: Apply column alignment classes after table render
            document.addEventListener('shiny:value', function(ev) {
                if (ev.detail && ev.detail.name === 'results_table') {
                    setTimeout(() => {
                        const table = document.querySelector('#results_container table');
                        if (!table) return;
                        
                        // Apply compact mode if needed
                        const compact = window.Shiny && Shiny.shinyapp && Shiny.shinyapp.$inputValues['compact_density'];
                        const container = document.querySelector('#results_container');
                        if (container) container.classList.toggle('compact', !!compact);
                        
                        // Fix column alignment
                        const headerCells = Array.from(table.querySelectorAll('thead th'));
                        const bodyRows = Array.from(table.querySelectorAll('tbody tr'));
                        
                        // Find the status column index
                        let statusColIdx = -1;
                        headerCells.forEach((th, idx) => {
                            const header = th.innerText || '';
                            if (header.toLowerCase() === 'status') {
                                statusColIdx = idx;
                            }
                        });
                        
                        headerCells.forEach((th, idx) => {
                            const header = th.innerText || '';
                            const isNum = /score|similarity|count|percent|^\\d+/.test(header.toLowerCase());
                            th.classList.toggle('num', isNum);
                            th.classList.toggle('text', !isNum);
                            
                            bodyRows.forEach(tr => {
                                const td = tr.children[idx];
                                if (!td) return;
                                const txt = td.innerText || '';
                                const looksNum = isNum || /^[\\d\\s\\.\\,\\-]+%?$/.test(txt);
                                td.classList.toggle('num', looksNum);
                                td.classList.toggle('text', !looksNum);
                                
                                // Add scorebar class for bar columns
                                if (/bar$/i.test(header)) {
                                    td.classList.add('scorebar');
                                }
                            });
                        });
                        
                        // Apply no-match-row class to rows with NO MATCH status
                        if (statusColIdx >= 0) {
                            bodyRows.forEach(tr => {
                                const statusCell = tr.children[statusColIdx];
                                if (statusCell && statusCell.innerText.trim() === 'NO MATCH') {
                                    tr.classList.add('no-match-row');
                                }
                            });
                        }
                    }, 100);
                }
            });

            // Ensure Step 1 preview tables stick left and first column stays narrow
            function adjustPreview(id){
              var root = document.getElementById(id);
              if(!root) return;
              // Remove auto-centering and allow full width on wrappers
              var nodes = [root].concat(Array.from(root.querySelectorAll('.shiny-data-frame, .html-fill-container, .html-fill-item, .table-responsive, table, div')));
              nodes.forEach(function(el){
                try {
                  el.style.marginLeft = '0';
                  el.style.marginRight = '0';
                  el.style.width = '100%';
                  el.style.maxWidth = 'none';
                } catch(e){}
              });
              var table = root.querySelector('table');
              if(!table) return;
              table.style.width = '100%';
              table.style.tableLayout = 'auto';
              // Add/adjust colgroup for first column
              var colgroup = table.querySelector('colgroup');
              var colCount = (table.querySelectorAll('thead th').length) || (table.querySelectorAll('tbody tr:first-child td').length);
              if(!colgroup && colCount){
                colgroup = document.createElement('colgroup');
                for (var i=0;i<colCount;i++){ colgroup.appendChild(document.createElement('col')); }
                table.insertBefore(colgroup, table.firstChild);
              }
              if(colgroup && colgroup.firstElementChild){
                var c0 = colgroup.firstElementChild;
                c0.style.width = '36px';
                c0.style.minWidth = '36px';
                c0.style.maxWidth = '40px';
              }
              // Set widths on first TH/TDs as well
              var th0 = table.querySelector('thead th:first-child');
              if(th0){ th0.style.width = '36px'; th0.style.minWidth = '36px'; th0.style.whiteSpace = 'nowrap'; th0.style.textAlign = 'center'; }
              table.querySelectorAll('tbody td:first-child').forEach(function(td){
                td.style.width = '36px';
                td.style.minWidth = '36px';
                td.style.whiteSpace = 'nowrap';
                td.style.textAlign = 'center';
              });
            }
            document.addEventListener('shiny:value', function(ev){
              if (ev.detail && (ev.detail.name === 'input_col_preview' || ev.detail.name === 'target_col_preview')){
                setTimeout(function(){ adjustPreview(ev.detail.name); }, 0);
              }
            });
            document.addEventListener('DOMContentLoaded', function(){
              adjustPreview('input_col_preview');
              adjustPreview('target_col_preview');
            });
        """)
        ,
        ui.tags.script("""
          (function(){
            // Debounced search mirror -> search_debounced
            var _t;
            document.addEventListener('input', function(e){
              if (e.target && e.target.id === 'search_filter'){
                clearTimeout(_t);
                var v = e.target.value;
                _t = setTimeout(function(){
                  if (window.Shiny && Shiny.setInputValue){
                    Shiny.setInputValue('search_debounced', v, {priority:'event'});
                  }
                }, 200);
              }
            });
          })();
        """)
    ),
    ui.navset_bar(
            ui.nav_panel(
                "Data & Configure",
                ui.card(
                    ui.card_header("Matching Setup"),
                    ui.card_body(
                        ui.row(
                            ui.column(6,
                                ui.input_select("input_column", "Input Column:", choices=[], selected=None),
                                ui.output_table("input_col_preview")
                            ),
                            ui.column(6,
                                ui.input_select("target_column", "Target Column:", choices=[], selected=None),
                                ui.output_table("target_col_preview")
                            )
                        ),
                        ui.hr(),
                        ui.row(
                            ui.column(6,
                                ui.p(ui.strong("Method:"), " Semantic Embedding (", ui.a("thenlper/gte-large", href="https://huggingface.co/thenlper/gte-large", target="_blank"), ")")
                            ),
                            ui.column(3,
                                ui.div(
                                    ui.span(
                                        "Similarity Threshold ",
                                        ui.tags.i(
                                            class_="bi bi-question-circle text-muted",
                                            **{"data-bs-toggle": "tooltip", "data-bs-placement": "top", "title": "Items below this score are marked as NO MATCH"}
                                        )
                                    ),
                                    ui.input_slider("threshold", "", min=0.0, max=1.0, value=0.85, step=0.05)
                                )
                            ),
                            ui.column(3,
                                ui.input_checkbox("clean_text", "Apply text cleaning", value=False)
                            )
                        ),
                        ui.br(),
                        ui.input_action_button("run_matching", "Start Mapping", class_="btn btn-primary", style="padding: 0.4rem 1rem;")
                    )
                ),
                ui.br(),
                ui.output_ui("center_progress_overlay")
            ),
            ui.nav_panel(
                "Results",
                ui.div(
                    ui.div(id="process_status"),
                    ui.div(id="process_progress"),
                    ui.div(id="process_summary"),
                    ui.div(id="method_chips"),
                    style="margin-bottom: 12px;"
                ),
                ui.card(
                    ui.card_header("Filter & View"),
                    ui.card_body(
                        ui.row(
                            ui.column(6, ui.input_text("search_filter", "Search:", placeholder="Type to filter...")),
                            ui.column(6,
                                ui.div(
                                    ui.input_checkbox("show_no_match", "Only NO MATCH", value=False),
                                    ui.input_checkbox("sort_by_score", "Sort by score", value=True),
                                    ui.input_checkbox("compact_density", "Compact", value=False),
                                    ui.input_checkbox("use_grid", "Interactive grid", value=True),
                                    class_="d-flex gap-3 flex-wrap justify-content-end"
                                )
                            ),
                        )
                    )
                ),
                ui.output_ui("results_tabulator"),
                ui.div(ui.output_table("results_table"), class_="results-container", id="results_container"),
                ui.br(),
                ui.card(
                    ui.card_header("Export"),
                    ui.card_body(
                        ui.div(
                            ui.download_button("download_results", ui.span(ui.tags.i(class_="bi bi-download me-2"), "Export All Results"), class_="btn btn-success"),
                            ui.download_button("download_no_match", ui.span(ui.tags.i(class_="bi bi-exclamation-triangle me-2"), "Export NO MATCH Only"), class_="btn btn-outline-warning"),
                            ui.input_action_button("reset_analysis", ui.span(ui.tags.i(class_="bi bi-arrow-repeat me-2"), "Start New Matching"), class_="btn btn-primary"),
                            class_="btn-group-export"
                        )
                    )
                )
            ),
            ui.nav_panel(
                "Visualizations",
                ui.card(
                    ui.card_header("Similarity Score Distribution"),
                    ui.card_body(
                        ui.row(
                            ui.column(6, ui.input_select("viz_type", "Chart", choices={"hist":"Histogram","cdf":"Cumulative","threshold":"Threshold curve","match_pie":"Matches vs No Matches"}, selected="hist")),
                        ),
                        ui.output_plot("score_hist")
                    )
                )
            ),
            ui.nav_panel(
                "About",
                ui.card(
                    ui.card_header("About FoodMapper"),
                    ui.card_body(
                        ui.h4("FoodMapper", class_="mb-3"),
                        ui.p("A state-of-the-art tool for matching food descriptions across different databases using advanced natural language processing techniques.", class_="lead"),
                        ui.hr(),
                        ui.h5(ui.tags.i(class_="bi bi-stars me-2"), "Key Features"),
                        ui.tags.ul(
                            ui.tags.li(ui.HTML("Semantic embedding via <a href=\"https://huggingface.co/thenlper/gte-large\" target=\"_blank\">thenlper/gte-large</a>")),
                            ui.tags.li("Batch processing with progress tracking"),
                            ui.tags.li("Clear results table with spreadsheet-like view"),
                            ui.tags.li("Export matched results and NO MATCH items")
                        )
                    )
                )
            ),
            id="main_tabs",
            title="FoodMapper"
        ),
    ui.br(),
    ui.div(
        ui.strong("Western Human Nutrition Research Center"),
        " | Davis, CA",
        ui.br(),
        "Diet, Microbiome and Immunity Research Unit",
        ui.br(),
        "United States Department of Agriculture | Agricultural Research Service",
    ),
    theme=shinyswatch.theme.yeti()
)
'''  # end legacy layout

def make_sidebar():
    return ui.sidebar(
        ui.div(
            ui.h5(
                "Upload Your Data",
                ui.input_action_link(
                    "show_upload_help",
                    ui.tags.i(class_="bi bi-question-circle ms-2", style="font-size: 0.8rem;"),
                    class_="text-muted",
                    style="text-decoration: none;"
                ),
                class_="d-flex justify-content-between align-items-center"
            )
        ),
        ui.input_file("input_file", "Input CSV", accept=[".csv"], multiple=False),
        ui.div(id="input_status", class_="mt-2"),
        ui.input_file("target_file", "Target CSV", accept=[".csv"], multiple=False),
        ui.div(id="target_status", class_="mt-2"),
        # Add sample data button for easier mobile access
        ui.div(
            ui.input_action_button(
                "sidebar_sample_data",
                "Or use sample data",
                class_="btn btn-outline-primary btn-sm w-100 mt-3"
            ),
            class_="mb-3"
        ),
        ui.hr(),
        ui.output_ui("sidebar_results_summary_block"),
        ui.hr(),
        # Dynamic navigation button that changes based on current step
        ui.output_ui("sidebar_navigation_button"),
        open="open",
        id="main_sidebar",
    )

def make_footer():
    return ui.div(
        ui.strong("Western Human Nutrition Research Center"),
        " | Davis, CA",
        ui.br(),
        "Diet, Microbiome and Immunity Research Unit",
        ui.br(),
        "United States Department of Agriculture | Agricultural Research Service",
        class_="footer"
    )

# Rebuild app UI with top navbar and nested tabs structure
app_ui = ui.page_navbar(
    # Top navigation bar panels
    ui.nav_panel(
        "Semantic Embedder",
        ui.page_sidebar(
            make_sidebar(),
            ui.tags.style(custom_css_min),
            ui.navset_tab(
                ui.nav_panel(
                    "Tutorial",
                    ui.card(
                        ui.card_header("Get Started"),
                        ui.card_body(
                            ui.row(
                                ui.column(6,
                                    ui.h6("Quick Start", class_="mb-2"),
                                    ui.tags.ul(
                                        ui.tags.li("Upload Input CSV"),
                                        ui.tags.li("Upload Target CSV"),
                                        ui.tags.li("Pick columns → Start")
                                    , class_="mb-2"),
                                    ui.tags.small(ui.tags.i(class_="bi bi-upload me-1"), "Use the sidebar on the left to add your files.", class_="text-muted d-block mb-2"),
                                    ui.div(
                                        ui.tags.small("No data? ", class_="text-muted"),
                                        ui.input_action_button("load_sample", "Try with sample data", class_="btn btn-outline-primary btn-sm"),
                                        class_="mt-2 mb-3"
                                    )
                                ),
                                ui.column(6,
                                    ui.h6(
                                        ui.span("Data Requirements"),
                                        ui.input_action_link(
                                            "show_requirements",
                                            ui.tags.i(class_="bi bi-info-circle ms-2"),
                                            class_="text-primary"
                                        ),
                                        class_="mb-2"
                                    ),
                                    ui.tags.ul(
                                        ui.tags.li("CSV files with headers"),
                                        ui.tags.li("Input: items to match"),
                                        ui.tags.li("Target: reference list")
                                    , class_="mb-2")
                                )
                            , class_="g-4 align-items-start"),
                            ui.hr(class_="my-3"),
                            ui.div(
                                ui.h6("What This Tool Does", class_="mb-2"),
                                ui.p(
                                    "This application matches text descriptions between two datasets using AI-powered semantic analysis. "
                                    "Upload your input items and target reference list, select the columns to match, "
                                    "and the tool will find the best semantic matches based on meaning rather than exact text.",
                                    class_="text-muted small"
                                ),
                                ui.h6("Key Features", class_="mb-2 mt-3"),
                                ui.tags.ul(
                                    ui.tags.li("Semantic matching using state-of-the-art embeddings", class_="small text-muted"),
                                    ui.tags.li("Adjustable similarity threshold for fine-tuning", class_="small text-muted"),
                                    ui.tags.li("Interactive visualizations and data export", class_="small text-muted"),
                                    ui.tags.li("Text cleaning options for better matches", class_="small text-muted")
                                ),
                                ui.div(id="sample_status", class_="mt-2")
                            )
                        )
                    )
                ),
                ui.nav_panel(
                    "Step 1: Data & Configure",
                ui.card(
                    ui.card_header("Matching Setup"),
                    ui.card_body(
                        ui.row(
                            ui.column(6,
                                ui.input_select("input_column", "Input Column:", choices=[], selected=None),
                                ui.input_switch("clean_input", "Apply text cleaning to input", value=False),
                                ui.div(
                                    ui.output_data_frame("input_col_preview"),
                                    id="input_preview_wrap",
                                    style="margin: 0 !important; padding: 0 !important; text-align: left !important; width: 100% !important;"
                                )
                            ),
                            ui.column(6,
                                ui.input_select("target_column", "Target Column:", choices=[], selected=None),
                                ui.input_switch("clean_target", "Apply text cleaning to target", value=False),
                                ui.div(
                                    ui.output_data_frame("target_col_preview"),
                                    id="target_preview_wrap",
                                    style="margin: 0 !important; padding: 0 !important; text-align: left !important; width: 100% !important;"
                                )
                            )
                        ),
                        ui.hr(),
                        ui.div(
                            ui.h5("Similarity Threshold", class_="text-center mb-2"),
                            
                            # Three-column layout for professional appearance (centered slider and button)
                            ui.row(
                                # Left column - Method info
                                ui.column(3,
                                    ui.div(
                                        ui.div(
                                            ui.tags.i(class_="bi bi-cpu text-primary me-1"),
                                            ui.strong("Method"),
                                            class_="mb-1 text-center small"
                                        ),
                                        ui.div("Semantic Embedding", class_="text-muted mb-2 text-center small"),
                                        ui.div(
                                            ui.tags.i(class_="bi bi-diagram-3 text-primary me-1"),
                                            ui.strong("Model"),
                                            class_="mb-1 text-center small"
                                        ),
                                        ui.div(
                                            ui.a("thenlper/gte-large", 
                                                href="https://huggingface.co/thenlper/gte-large", 
                                                target="_blank",
                                                class_="text-primary small"),
                                            class_="text-center small"
                                        ),
                                        class_="border-end pe-3 py-2"
                                    )
                                ),
                                # Center column - Threshold slider and button stacked and centered
                                ui.column(6,
                                    ui.div(
                                        ui.div(
                                            ui.div(
                                                ui.input_slider("threshold", "", min=0.0, max=1.0, value=0.85, step=0.01),
                                                class_="mx-auto",
                                                style="max-width: 400px;"
                                            ),
                                            class_="d-flex justify-content-center mb-1"
                                        ),
                                        ui.div(
                                            ui.input_action_button(
                                                "run_matching", 
                                                "Start Mapping", 
                                                class_="btn btn-primary px-5 shadow-sm",
                                                style="padding: 0.5rem 2rem;"
                                            ),
                                            class_="text-center mt-2"
                                        )
                                    )
                                ),
                                # Right column - Threshold note
                                ui.column(3,
                                    ui.div(
                                        ui.div(
                                            ui.tags.i(class_="bi bi-info-circle text-muted me-1"),
                                            ui.tags.small(ui.strong("Note"), class_="text-muted"),
                                            class_="mb-1"
                                        ),
                                        ui.tags.small(
                                            ui.div("Items below threshold", class_="text-muted lh-sm"),
                                            ui.div("marked as NO MATCH.", class_="text-muted lh-sm"),
                                            ui.div("Adjust for performance", class_="text-muted lh-sm mt-1"),
                                            ui.div("for your dataset.", class_="text-muted lh-sm")
                                        ),
                                        class_="border-start ps-3 py-2"
                                    )
                                )
                            ),
                            
                            class_="matching-config-card"
                        )
                    )
                ),
                ui.br(),
                ui.output_ui("center_progress_overlay")
            ),
            ui.nav_panel(
                "Step 2: Results",
                ui.navset_pill(
                    ui.nav_panel(
                        "View Mappings",
                ui.div(
                    ui.div(id="process_status"),
                    ui.div(id="process_progress"),
                    ui.div(id="process_summary"),
                    ui.div(id="method_chips"),
                    style="margin-bottom: 12px;"
                ),
                ui.output_ui("results_tabulator"),
                # Add tip below the results table with clickable link
                ui.div(
                    ui.tags.small(
                        ui.tags.i(class_="bi bi-lightbulb me-1"),
                        "Tip: Return to ",
                        ui.input_action_link("goto_step1_from_tip", "Step 1: Data & Configure", class_="text-primary"),
                        " to adjust threshold or column selections, then re-run mapping.",
                        class_="text-muted"
                    ),
                    class_="mt-3 mb-3 text-center"
                ),
                ui.div(
                    ui.download_button("download_all_data", ui.span(ui.tags.i(class_="bi bi-download me-2"), "Export All Data"), class_="btn btn-success"),
                    ui.download_button("download_matches", ui.span(ui.tags.i(class_="bi bi-file-earmark-check me-2"), "Export Matches"), class_="btn btn-info"),
                    ui.input_action_button("reset_analysis", ui.span(ui.tags.i(class_="bi bi-arrow-repeat me-2"), "Start New Mapping"), class_="btn btn-primary"),
                    class_="btn-group-export"
                )
            ),
            ui.nav_panel(
                "Visualizations",
                ui.card(
                    ui.card_header("Interactive Visualizations"),
                    ui.card_body(
                        ui.row(
                            ui.column(7,
                                ui.input_select("plotly_viz_type", "Chart Type",
                                    choices={
                                        "density": "Density Plot - Score distribution shape",
                                        "histogram": "Histogram - Frequency of score ranges",
                                        "threshold": "Threshold Analysis - Match rate at different cutoffs"
                                        # HIDDEN VISUALIZATIONS - Uncomment lines below to restore
                                        # "box": "Box Plot - Quartiles & outliers",
                                        # "violin": "Violin Plot - Match vs No-match comparison",
                                        # "scatter": "Scatter Plot - Sequential patterns",
                                        # "ecdf": "Cumulative Distribution - Probability curve",
                                        # "sunburst": "Match Breakdown - Hierarchical match statistics"
                                    },
                                    selected="density",
                                    width="100%"
                                )
                            ),
                            ui.column(5,
                                ui.input_checkbox("show_threshold_line", "Show threshold line", value=True)
                            )
                        ),
                        ui.output_ui("chart_description"),
                        output_widget("plotly_viz")
                    )
                )
            ),
            id="results_subtabs"
        )
    ),
    id="workflow_tabs"
),
            # JS helpers for Step 1 preview layout and narrow first column
            ui.tags.script("""
              (function(){
                // Initialize Bootstrap tooltips on demand
                function initTooltips(){
                  if (window.bootstrap && bootstrap.Tooltip) {
                    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(function(el){
                      try { new bootstrap.Tooltip(el, {container:'body'}); } catch(e){}
                    });
                  }
                }
                document.addEventListener('DOMContentLoaded', initTooltips);
                document.addEventListener('shiny:value', initTooltips);
                // Threshold value badge removed; rely on slider only
                function adjustPreview(id){
                  var root = document.getElementById(id);
                  if(!root) return;
                  // Ensure containers don't center and take full width
                  var containers = root.querySelectorAll('.shiny-data-frame, .html-fill-container, .html-fill-item, .gridjs-container, .gridjs-wrapper');
                  containers.forEach(function(el){
                    try {
                      el.style.marginLeft = '0';
                      el.style.marginRight = '0';
                      el.style.width = '100%';
                      el.style.maxWidth = 'none';
                    } catch(e){}
                  });
                  var table = root.querySelector('.gridjs-table');
                  if(!table) return;
                  table.style.width = '100%';
                  table.style.tableLayout = 'auto';
                  // Force first column narrow across header and body (Grid.js)
                  var th0 = table.querySelector('thead.gridjs-thead th.gridjs-th:first-child');
                  if (th0) {
                    th0.style.setProperty('width','36px','important');
                    th0.style.setProperty('min-width','36px','important');
                    th0.style.setProperty('max-width','40px','important');
                    th0.style.setProperty('white-space','nowrap','important');
                    th0.style.setProperty('text-align','center','important');
                  }
                  table.querySelectorAll('tbody.gridjs-tbody td.gridjs-td:first-child').forEach(function(td){
                    td.style.setProperty('width','36px','important');
                    td.style.setProperty('min-width','36px','important');
                    td.style.setProperty('max-width','40px','important');
                    td.style.setProperty('white-space','nowrap','important');
                    td.style.setProperty('text-align','center','important');
                  });
                }
                document.addEventListener('shiny:value', function(ev){
                  if (ev.detail && (ev.detail.name === 'input_col_preview' || ev.detail.name === 'target_col_preview')){
                    setTimeout(function(){ adjustPreview(ev.detail.name); }, 0);
                  }
                });
                document.addEventListener('DOMContentLoaded', function(){
                  adjustPreview('input_col_preview');
                  adjustPreview('target_col_preview');
                });
              })();
            """),
            make_footer()
        )
    ),
    ui.nav_panel(
        "About",
        ui.div(
            ui.card(
                ui.card_header(
                    ui.h4("About FoodMapper", class_="mb-0")
                ),
                ui.card_body(
                    # Hero Section
                    ui.div(
                        ui.h2("FoodMapper", class_="text-center mb-3"),
                        ui.p(
                            "Advanced semantic matching tool for aligning food descriptions across nutritional databases",
                            class_="lead text-center text-muted mb-4"
                        ),
                        ui.hr(class_="my-4")
                    ),
                    
                    # Overview Section
                    ui.div(
                        ui.h5(
                            ui.tags.i(class_="bi bi-info-circle me-2"),
                            "Overview",
                            class_="mb-3"
                        ),
                        ui.p(
                            "FoodMapper solves a major problem in nutritional research: accurately matching "
                            "food items between different databases that use varying naming conventions and descriptions. "
                            "This tool uses neural language processing to find semantic matches "
                            "based on meaning rather than exact text matching.",
                            class_="mb-4"
                        )
                    ),
                    
                    # Problem Statement
                    ui.div(
                        ui.h5(
                            ui.tags.i(class_="bi bi-question-circle me-2"),
                            "The Challenge",
                            class_="mb-3 mt-4"
                        ),
                        ui.p(
                            "Nutritional databases often describe the same foods differently:",
                            class_="mb-2"
                        ),
                        ui.tags.ul(
                            ui.tags.li('"2% milk" vs "Milk, reduced fat, 2% milkfat"'),
                            ui.tags.li('"OJ" vs "Orange juice, raw"'),
                            ui.tags.li('"Whole wheat bread" vs "Bread, whole-wheat, commercially prepared"'),
                            class_="mb-3"
                        ),
                        ui.p(
                            "Traditional text matching fails to recognize these as the same items, leading to "
                            "incomplete or inaccurate nutritional analyses.",
                            class_="text-muted mb-4"
                        )
                    ),
                    
                    # Solution Section
                    ui.div(
                        ui.h5(
                            ui.tags.i(class_="bi bi-lightbulb me-2"),
                            "Our Solution",
                            class_="mb-3 mt-4"
                        ),
                        ui.p(
                            "FoodMapper uses semantic embeddings to understand the meaning behind food descriptions, "
                            "enabling accurate matches even when the exact wording differs.",
                            class_="mb-3"
                        ),
                        ui.div(
                            ui.row(
                                ui.column(6,
                                    ui.div(
                                        ui.tags.i(class_="bi bi-cpu text-primary fs-3 mb-2 d-block"),
                                        ui.h6("AI Model", class_="mb-2"),
                                        ui.p(
                                            ui.HTML('Powered by <a href="https://huggingface.co/thenlper/gte-large" target="_blank" class="text-decoration-none">GTE-Large</a>'),
                                            ui.br(),
                                            ui.tags.small("Neural embedding model", class_="text-muted"),
                                            class_="small"
                                        ),
                                        class_="text-center p-3 border rounded mb-3"
                                    )
                                ),
                                ui.column(6,
                                    ui.div(
                                        ui.tags.i(class_="bi bi-speedometer2 text-success fs-3 mb-2 d-block"),
                                        ui.h6("Performance", class_="mb-2"),
                                        ui.p(
                                            "Process thousands of items/minute",
                                            ui.br(),
                                            ui.tags.small("Batch processing system", class_="text-muted"),
                                            class_="small"
                                        ),
                                        class_="text-center p-3 border rounded mb-3"
                                    )
                                )
                            ),
                            ui.row(
                                ui.column(6,
                                    ui.div(
                                        ui.tags.i(class_="bi bi-bullseye text-info fs-3 mb-2 d-block"),
                                        ui.h6("Accuracy", class_="mb-2"),
                                        ui.p(
                                            "Semantic understanding",
                                            ui.br(),
                                            ui.tags.small("Matches based on meaning", class_="text-muted"),
                                            class_="small"
                                        ),
                                        class_="text-center p-3 border rounded mb-3"
                                    )
                                ),
                                ui.column(6,
                                    ui.div(
                                        ui.tags.i(class_="bi bi-sliders text-warning fs-3 mb-2 d-block"),
                                        ui.h6("Control", class_="mb-2"),
                                        ui.p(
                                            "Adjustable thresholds",
                                            ui.br(),
                                            ui.tags.small("Fine-tune match sensitivity", class_="text-muted"),
                                            class_="small"
                                        ),
                                        class_="text-center p-3 border rounded mb-3"
                                    )
                                )
                            )
                        )
                    ),
                    
                    # Key Features
                    ui.div(
                        ui.h5(
                            ui.tags.i(class_="bi bi-star me-2"),
                            "Key Features",
                            class_="mb-3 mt-4"
                        ),
                        ui.tags.ul(
                            ui.tags.li(
                                ui.strong("Semantic Matching:"),
                                " Understands food descriptions using neural embeddings"
                            ),
                            ui.tags.li(
                                ui.strong("Batch Processing:"),
                                " Handle thousands of items efficiently with concurrent processing"
                            ),
                            ui.tags.li(
                                ui.strong("Interactive Visualizations:"),
                                " Explore match distributions and patterns with 8 chart types"
                            ),
                            ui.tags.li(
                                ui.strong("Data Export:"),
                                " Download results as CSV with all original data preserved"
                            ),
                            ui.tags.li(
                                ui.strong("Text Cleaning:"),
                                " Optional preprocessing to potentially improve match quality"
                            ),
                            ui.tags.li(
                                ui.strong("Real-time Preview:"),
                                " See data transformations before processing"
                            ),
                            class_="mb-4"
                        )
                    ),
                    
                    # Use Cases
                    ui.div(
                        ui.h5(
                            ui.tags.i(class_="bi bi-diagram-3 me-2"),
                            "Use Cases",
                            class_="mb-3 mt-4"
                        ),
                        ui.tags.ul(
                            ui.tags.li("Harmonizing dietary intake data with nutrient databases"),
                            ui.tags.li("Linking research datasets to food composition tables"),
                            ui.tags.li("Standardizing food nomenclature across studies"),
                            ui.tags.li("Quality control for nutritional data entry"),
                            ui.tags.li("Cross-referencing international food databases"),
                            class_="mb-4"
                        )
                    ),
                    
                    # Credits Section
                    ui.div(
                        ui.hr(class_="my-4"),
                        ui.h5(
                            ui.tags.i(class_="bi bi-people me-2"),
                            "Development Team",
                            class_="mb-3"
                        ),
                        ui.div(
                            ui.p(
                                ui.strong("Principal Investigator:"),
                                " Dr. Danielle G. Lemay",
                                ui.br(),
                                ui.tags.small("Research Molecular Biologist", class_="text-muted"),
                                class_="mb-2"
                            ),
                            ui.p(
                                ui.strong("Developer:"),
                                " Richard Stoker",
                                ui.br(),
                                ui.tags.small("IT Specialist (Scientific)", class_="text-muted"),
                                class_="mb-2"
                            ),
                            ui.p(
                                ui.strong("Organization:"),
                                ui.br(),
                                "USDA Agricultural Research Service",
                                ui.br(),
                                "Western Human Nutrition Research Center",
                                ui.br(),
                                ui.tags.small("Davis, California", class_="text-muted"),
                                class_="mb-3"
                            )
                        )
                    ),
                    
                    # Version and Contact
                    ui.div(
                        ui.hr(class_="my-4"),
                        ui.row(
                            ui.column(6,
                                ui.p(
                                    ui.tags.i(class_="bi bi-tag me-1"),
                                    ui.strong("Version:"),
                                    " 1.0.0",
                                    class_="text-muted small mb-0"
                                )
                            ),
                            ui.column(6,
                                ui.p(
                                    ui.tags.i(class_="bi bi-envelope me-1"),
                                    ui.strong("Contact:"),
                                    " richard.stoker@usda.gov",
                                    ui.br(),
                                    ui.HTML('<a href="https://github.com/RichardStoker-USDA/Semantic-Food-Mapping-ShinyApp" target="_blank" class="text-decoration-none"><i class="bi bi-github me-1"></i>GitHub</a>'),
                                    class_="text-muted small mb-0 text-end"
                                )
                            )
                        )
                    )
                )
            ),
            make_footer(),
            class_="container-fluid",
            style="max-width: 1200px; margin: 0 auto; padding: 20px;"
        )
    ),
    title=ui.tags.div(
        ui.tags.h2(
            "FoodMapper", 
            style="margin: 0 1rem 0 0; font-size: 1.9rem; font-weight: 700; letter-spacing: 0.5px; padding-right: 1rem; border-right: 2px solid var(--bs-gray-400, #ced4da);"
        ),
        ui.tags.div(
            ui.input_dark_mode(id="dark_mode", mode="light"),
            style="margin-left: 1rem;"
        ),
        style="display: flex; align-items: center;"
    ),
    id="main_navbar",
    # Remove hardcoded bg color to let theme control it
    theme=shinyswatch.theme.yeti()
)

# UI utility functions
def create_score_bar(score: float, width: int = 12, min_scale: float = 0.5) -> str:
    """Create a text-based progress bar for score visualization
    
    For semantic embeddings, scales the bar relative to a minimum threshold
    since scores rarely go below 0.5, making differences more visible.
    """
    try:
        score = 0.0 if score is None or pd.isna(score) else float(score)
    except:
        score = 0.0
    
    # For semantic embeddings, scale relative to min_scale (default 0.5)
    # This makes the bar show relative differences better
    if score < min_scale:
        # If below min_scale, show as empty or minimal
        filled = 0
    else:
        # Scale from min_scale to 1.0 across the full bar width
        scaled_score = (score - min_scale) / (1.0 - min_scale)
        scaled_score = max(0.0, min(1.0, scaled_score))
        filled = int(round(scaled_score * width))
    
    return "█" * filled + "░" * (width - filled)

def create_status_badge(value: str) -> str:
    """Create HTML status badge based on match status"""
    if str(value).upper() == "NO MATCH":
        return '<span class="status-badge status-warn">NO MATCH</span>'
    else:
        return '<span class="status-badge status-ok">Match</span>'

def server(input: Inputs, output: Outputs, session: Session):
    # Reactive values for data storage
    input_df = reactive.value(pd.DataFrame())
    target_df = reactive.value(pd.DataFrame())
    results_df = reactive.value(pd.DataFrame())
    current_step = reactive.value(1)
    progress_message = reactive.value("Starting...")
    # State for optional centered overlay (currently returns None by default)
    progress_show = reactive.value(False)
    progress_percent = reactive.value(0)
    
    # Show welcome splash screen on app load (controlled by environment variable)
    @reactive.effect
    def show_splash():
        # Check environment variable to control splash screen visibility
        # HIDE_SPLASH_SCREEN: Set to "true" or "1" to hide the splash screen
        # Default behavior is to show the splash screen if variable is not set
        hide_splash = os.environ.get("HIDE_SPLASH_SCREEN", "").lower() in ["true", "1", "yes"]
        
        if not hide_splash:
            ui.modal_show(
                ui.modal(
                    ui.div(
                        # Header
                        ui.h2("FoodMapper", class_="text-center mb-2"),
                        ui.p(
                            "Research Tool for Dietary Data Mapping",
                            class_="text-center text-muted mb-4"
                        ),
                        ui.hr(),
                        
                        # Research Paper Section
                        ui.div(
                            ui.h5(
                                ui.tags.i(class_="bi bi-journal-text me-2"),
                                "Research Publication",
                                class_="mb-3"
                            ),
                            ui.div(
                                ui.p(
                                    "This application was developed as part of ongoing research on automated methods "
                                    "for mapping dietary intake data to food composition databases.",
                                    class_="mb-3"
                                ),
                                ui.div(
                                    ui.p(
                                        ui.strong("Paper Title:"),
                                        ui.br(),
                                        ui.tags.small(
                                            "[Title Placeholder - To Be Updated]",
                                            class_="text-muted"
                                        ),
                                        class_="mb-2"
                                    ),
                                    ui.p(
                                        ui.strong("Authors:"),
                                        ui.br(),
                                        ui.tags.small(
                                            "Lemay DG, Strohmeier MP, Stoker RB, Larke JA, Wilson SMG",
                                            class_="text-muted"
                                        ),
                                        class_="mb-2"
                                    ),
                                    ui.p(
                                        ui.strong("Learn More:"),
                                        ui.br(),
                                        ui.tags.small(
                                            ui.HTML('[<a href="#" target="_blank">Link to paper - Coming Soon</a>]'),
                                            class_="text-muted"
                                        ),
                                        class_="mb-2"
                                    ),
                                    ui.p(
                                        ui.HTML('<a href="https://github.com/RichardStoker-USDA/Semantic-Food-Mapping-ShinyApp" target="_blank" class="text-decoration-none text-muted"><i class="bi bi-github"></i> View on GitHub</a>'),
                                        class_="mb-3 small"
                                    ),
                                    class_="ms-3 border-start ps-3"
                                )
                            ),
                            class_="mb-4"
                        ),
                        
                        # How It Works Section
                        ui.div(
                            ui.h5(
                                ui.tags.i(class_="bi bi-info-circle me-2"),
                                "How It Works",
                                class_="mb-3"
                            ),
                            ui.p(
                                "FoodMapper uses the GTE-Large neural embedding model to understand the meaning "
                                "behind food descriptions. This enables accurate matching even when foods are described "
                                "differently across databases.",
                                class_="small mb-2"
                            ),
                            ui.p(
                                "Traditional manual mapping takes ~28 minutes per food item. "
                                "This tool automates the process, handling thousands of items in minutes.",
                                class_="small text-muted"
                            ),
                            class_="mb-4"
                        ),
                        
                        # Disclaimer
                        ui.div(
                            ui.hr(),
                            ui.p(
                                ui.tags.i(class_="bi bi-exclamation-triangle me-1"),
                                ui.strong("Research Tool Disclaimer"),
                                class_="text-center mb-2"
                            ),
                            ui.p(
                                "This application is a research tool intended for scientific use in nutritional and dietary studies. "
                                "Results should be validated by domain experts. For research purposes only.",
                                class_="small text-muted text-center"
                            ),
                            class_="mt-3"
                        ),
                        
                        # Get Started Button
                        ui.div(
                            ui.input_action_button(
                                "close_splash", 
                                "Get Started", 
                                class_="btn btn-primary btn-lg"
                            ),
                            class_="text-center mt-4"
                        ),
                        class_="p-4"
                    ),
                    title="",
                    footer=None,
                    size="m",
                    easy_close=True,
                    fade=True
                )
            )
    
    # Close splash screen handler
    @reactive.effect
    @reactive.event(input.close_splash)
    def close_splash():
        ui.modal_remove()
    
    # Search input debouncing implementation
    @reactive.calc
    def debounced_search():
        try:
            return input.search_debounced()
        except Exception:
            return input.search_filter()

    # Threshold badge removed; using slider only
    
    # Dynamic sidebar navigation button
    @render.ui
    def sidebar_navigation_button():
        # Get current active tab - workflow_tabs tracks the main steps
        try:
            current_tab = input.workflow_tabs()
        except:
            current_tab = "Tutorial"  # Default to Step 0
        
        if current_tab == "Tutorial":
            # Check if both files are loaded
            in_df = input_df.get()
            tgt_df = target_df.get()
            files_loaded = (not in_df.empty) and (not tgt_df.empty)
            
            if files_loaded:
                return ui.input_action_button(
                    "sidebar_next", 
                    ui.span(ui.tags.i(class_="bi bi-arrow-right-circle-fill me-2"), "Next: Configure Data"),
                    class_="btn btn-primary w-100"
                )
            else:
                return ui.input_action_button(
                    "sidebar_next", 
                    ui.span(ui.tags.i(class_="bi bi-arrow-right-circle-fill me-2"), "Next: Configure Data"),
                    class_="btn btn-primary w-100",
                    disabled=True
                )
        elif current_tab == "Step 1: Data & Configure":
            # Check if results are available
            if not results_df.get().empty:
                return ui.input_action_button(
                    "sidebar_next",
                    ui.span(ui.tags.i(class_="bi bi-arrow-right-circle-fill me-2"), "Next: View Results"),
                    class_="btn btn-primary w-100"
                )
            else:
                return ui.input_action_button(
                    "sidebar_next",
                    ui.span(ui.tags.i(class_="bi bi-arrow-right-circle-fill me-2"), "Next: View Results"),
                    class_="btn btn-primary w-100 disabled",
                    disabled=True
                )
        elif current_tab == "Step 2: Results":
            return ui.input_action_button(
                "sidebar_reset",
                ui.span(ui.tags.i(class_="bi bi-arrow-counterclockwise me-2"), "Start New Mapping"),
                class_="btn btn-primary w-100"
            )
        else:
            return None
    
    
    # Sidebar navigation button handler
    @reactive.effect
    @reactive.event(input.sidebar_next)
    def handle_sidebar_next():
        try:
            current_tab = input.workflow_tabs()
        except:
            current_tab = "Tutorial"
        
        if current_tab == "Tutorial":
            # Only navigate if files are loaded
            in_df = input_df.get()
            tgt_df = target_df.get()
            if (not in_df.empty) and (not tgt_df.empty):
                ui.update_navs("workflow_tabs", selected="Step 1: Data & Configure")
        elif current_tab == "Step 1: Data & Configure":
            ui.update_navs("workflow_tabs", selected="Step 2: Results")
    
    # Sidebar reset button handler
    @reactive.effect
    @reactive.event(input.sidebar_reset)
    def handle_sidebar_reset():
        # Clear everything for a fresh start
        reset_for_new_analysis()
        
        # Navigate to Step 1 (not Step 0)
        ui.update_navs("workflow_tabs", selected="Step 1: Data & Configure")
        
        # Expand the sidebar so user can upload new files
        ui.update_sidebar("main_sidebar", show=True)
        
        # Show notification
        ui.notification_show(
            "Ready for new mapping. Upload your data files.",
            type="info",
            duration=3
        )

    # Load sample data from sidebar button (mobile-friendly)
    @reactive.effect
    @reactive.event(input.sidebar_sample_data)
    def load_sample_from_sidebar():
        # Load the sample data
        sample_input, sample_target = get_sample_data()
        input_df.set(sample_input)
        target_df.set(sample_target)
        
        # Update column choices
        ui.update_select("input_column", 
                        choices=sample_input.columns.tolist(),
                        selected="description")
        ui.update_select("target_column",
                        choices=sample_target.columns.tolist(),
                        selected="reference")
        
        # Enable buttons since data is loaded
        check_files_loaded()
        
        # Close the sidebar (important for mobile)
        ui.update_sidebar("main_sidebar", show=False)
        
        # Navigate to Step 1
        ui.update_navs("workflow_tabs", selected="Step 1: Data & Configure")
        
        # Show success notification
        ui.notification_show(
            "Sample data loaded! Configure your matching settings.",
            type="success",
            duration=3
        )
    
    # Load sample data from tutorial page
    @reactive.effect
    @reactive.event(input.load_sample)
    def load_sample_data():
        # Remove previous sample status message if present
        try:
            ui.remove_ui(selector="#sample_status_msg")
        except Exception:
            pass
        sample_input, sample_target = get_sample_data()
        input_df.set(sample_input)
        target_df.set(sample_target)
        
        # Update column choices
        ui.update_select("input_column", 
                        choices=sample_input.columns.tolist(),
                        selected="description")
        ui.update_select("target_column",
                        choices=sample_target.columns.tolist(),
                        selected="reference")
        
        # Show status
        ui.insert_ui(
            ui.div(
                ui.p("Sample data loaded successfully! Moving to Data Setup...", 
                     class_="alert alert-success alert-animated"),
                id="sample_status_msg"
            ),
            selector="#sample_status",
            where="afterEnd"
        )
        
        # Enable buttons since data is loaded
        check_files_loaded()
        
        # Automatically navigate to Step 1 after loading sample data
        ui.update_navs("workflow_tabs", selected="Step 1: Data & Configure")

    # Navigate to Step 1 from tip link
    @reactive.effect
    @reactive.event(input.goto_step1_from_tip)
    def goto_step1_from_tip():
        ui.update_navs("workflow_tabs", selected="Step 1: Data & Configure")
    
    # Show Upload Help modal when question icon clicked in sidebar
    @reactive.effect
    @reactive.event(input.show_upload_help)
    def show_upload_help_modal():
        ui.modal_show(
            ui.modal(
                ui.div(
                    ui.h4("Upload Requirements", class_="mb-3"),
                    ui.hr(),
                    ui.h6("File Format"),
                    ui.tags.ul(
                        ui.tags.li("CSV format (.csv) required"),
                        ui.tags.li("Include headers in first row"),
                        ui.tags.li("UTF-8 encoding recommended")
                    ),
                    ui.h6("Input File", class_="mt-3"),
                    ui.p("Items you want to match (one per row)", class_="text-muted"),
                    ui.h6("Target File", class_="mt-3"),
                    ui.p("Reference database to match against", class_="text-muted"),
                    ui.hr(),
                    ui.p(
                        ui.tags.small(
                            "Need help? ",
                            ui.input_action_link("close_help_goto_tutorial", "View tutorial", class_="text-primary"),
                            class_="text-muted"
                        )
                    ),
                    class_="p-2"
                ),
                footer=ui.input_action_button("close_upload_help", "Got it", class_="btn btn-primary"),
                easy_close=True,
                size="m",
                title=""
            )
        )
    
    # Close upload help modal
    @reactive.effect
    @reactive.event(input.close_upload_help)
    def close_upload_help():
        ui.modal_remove()
    
    # Close help and go to tutorial
    @reactive.effect
    @reactive.event(input.close_help_goto_tutorial)
    def close_help_goto_tutorial():
        ui.modal_remove()
        ui.update_sidebar("main_sidebar", show=False)
        ui.update_navs("workflow_tabs", selected="Tutorial")
    
    # Show Data Requirements modal when info icon clicked
    @reactive.effect
    @reactive.event(input.show_requirements)
    def show_data_requirements_modal():
        ui.modal_show(
            ui.modal(
                ui.div(
                    ui.h4("Data Requirements", class_="mb-3"),
                    ui.hr(),
                    ui.h6("File Format"),
                    ui.tags.ul(
                        ui.tags.li("Files must be in CSV format (.csv)"),
                        ui.tags.li("Must include column headers in first row"),
                        ui.tags.li("UTF-8 encoding recommended")
                    ),
                    ui.h6("Input File", class_="mt-3"),
                    ui.p("Contains the items you want to match. Each row represents one item to find a match for.", class_="text-muted"),
                    ui.h6("Target File", class_="mt-3"),
                    ui.p("Contains the reference dataset. The system will find the best match from this list for each input item.", class_="text-muted"),
                    ui.h6("Best Practices", class_="mt-3"),
                    ui.tags.ul(
                        ui.tags.li("Choose columns with descriptive text for best semantic matching"),
                        ui.tags.li("Remove or clean special characters if needed"),
                        ui.tags.li("Longer descriptions generally produce better matches")
                    ),
                    class_="p-2"
                ),
                footer=ui.input_action_button("close_req", "Got it", class_="btn btn-primary"),
                easy_close=True,
                size="m",
                title=""
            )
        )

    # Close Data Requirements modal
    @reactive.effect
    @reactive.event(input.close_req)
    def close_data_requirements_modal():
        ui.modal_remove()
    
    # Handle file uploads
    @reactive.effect
    @reactive.event(input.input_file)
    def handle_input_file():
        file: list[FileInfo] | None = input.input_file()
        if file and len(file) > 0:
            df = pd.read_csv(file[0]["datapath"])
            input_df.set(df)
            
            # Update column choices
            ui.update_select("input_column",
                           choices=df.columns.tolist(),
                           selected=df.columns[0])
            
            # Remove previous message then insert a fresh, readable status with filename
            try:
                ui.remove_ui(selector="#input_file_status")
            except Exception:
                pass
            ui.insert_ui(
                ui.div(
                    ui.p(f"Loaded {len(df):,} inputs",
                         class_="alert alert-success alert-animated"),
                    id="input_file_status"
                ),
                selector="#input_status",
                where="afterEnd"
            )
            
            # Check if both files loaded
            check_files_loaded()

    # NOTE: Clear files handler removed as button was removed from UI

    @reactive.effect
    @reactive.event(input.target_file)
    def handle_target_file():
        file: list[FileInfo] | None = input.target_file()
        if file and len(file) > 0:
            df = pd.read_csv(file[0]["datapath"])
            target_df.set(df)
            
            # Update column choices
            ui.update_select("target_column",
                           choices=df.columns.tolist(),
                           selected=df.columns[0])
            
            # Remove previous message then insert a fresh, readable status with filename
            try:
                ui.remove_ui(selector="#target_file_status")
            except Exception:
                pass
            ui.insert_ui(
                ui.div(
                    ui.p(f"Loaded {len(df):,} targets",
                         class_="alert alert-success alert-animated"),
                    id="target_file_status"
                ),
                selector="#target_status",
                where="afterEnd"
            )
            
            # Check if both files loaded
            check_files_loaded()
    
    # Preview tables
    @render.table
    def input_preview():
        df = input_df.get()
        if not df.empty:
            return df.head(5)
        return pd.DataFrame()
    
    @render.table
    def target_preview():
        df = target_df.get()
        if not df.empty:
            return df.head(5)
        return pd.DataFrame()
    
    @render.data_frame
    def input_col_preview():
        df = input_df.get()
        col = input.input_column()
        if not df.empty and col and col in df.columns:
            sample_values = df[col].dropna().head(5).tolist()
            
            # Apply text cleaning if toggle is on
            if input.clean_input():
                original_values = sample_values.copy()
                cleaned_values = clean_text_simple(sample_values)
                preview_df = pd.DataFrame({
                    "Row": range(1, len(sample_values) + 1),
                    "Original": original_values,
                    "After Cleaning": cleaned_values
                })
            else:
                preview_df = pd.DataFrame({
                    "Row": range(1, len(sample_values) + 1),
                    "Sample Values": sample_values
                })
            # Use Shiny DataGrid (theme-aware, interactive)
            return render.DataGrid(preview_df)
        return pd.DataFrame()
    
    @render.data_frame
    def target_col_preview():
        df = target_df.get()
        col = input.target_column()
        if not df.empty and col and col in df.columns:
            sample_values = df[col].dropna().head(5).tolist()
            
            # Apply text cleaning if toggle is on
            if input.clean_target():
                original_values = sample_values.copy()
                cleaned_values = clean_text_simple(sample_values)
                preview_df = pd.DataFrame({
                    "Row": range(1, len(sample_values) + 1),
                    "Original": original_values,
                    "After Cleaning": cleaned_values
                })
            else:
                preview_df = pd.DataFrame({
                    "Row": range(1, len(sample_values) + 1),
                    "Sample Values": sample_values
                })
            # Use Shiny DataGrid (theme-aware, interactive)
            return render.DataGrid(preview_df)
        return pd.DataFrame()
    
    # Helper function to check readiness for running mapping
    def check_files_loaded():
        ready = (not input_df.get().empty) and (not target_df.get().empty)
        try:
            in_col = input.input_column()
            tgt_col = input.target_column()
            ready = ready and bool(in_col) and bool(tgt_col)
        except Exception:
            pass
        ui.update_action_button("run_matching", disabled=(not ready))

    # Watch column selection to enable/disable run button
    @reactive.effect
    def _watch_columns_for_run():
        try:
            _ = (input.input_column(), input.target_column())
        except Exception:
            pass
        check_files_loaded()
    
    # Navigation handlers
    # Navigation effects removed in sidebar layout
    
    # Reset analysis handler from Results page button
    @reactive.effect
    @reactive.event(input.reset_analysis)
    def handle_reset_analysis():
        # Clear everything for a fresh start
        reset_for_new_analysis()
        
        # Navigate to Step 1 (not Step 0)
        ui.update_navs("workflow_tabs", selected="Step 1: Data & Configure")
        
        # Expand the sidebar so user can upload new files
        ui.update_sidebar("main_sidebar", show=True)
        
        # Show notification
        ui.notification_show(
            "Ready for new mapping. Upload your data files.",
            type="info",
            duration=3
        )
    
    def reset_for_new_analysis():
        # Clear results
        results_df.set(pd.DataFrame())
        # Clear input and target datasets and selections
        input_df.set(pd.DataFrame())
        target_df.set(pd.DataFrame())
        ui.update_select("input_column", choices=[], selected=None)
        ui.update_select("target_column", choices=[], selected=None)
        # Disable run button until files and columns are (re)selected
        ui.update_action_button("run_matching", disabled=True)
        # Clear file upload status messages
        for sel in ["#sample_status_msg", "#process_status > *", "#process_progress > *", "#process_summary > *", "#results_summary_msg", "#input_file_status", "#target_file_status"]:
            try:
                ui.remove_ui(selector=sel)
            except Exception:
                pass
    
    # Optional centered overlay (placeholder: disabled by default)
    @render.ui
    def center_progress_overlay():
        # Return None so nothing is rendered; keep hook for future use
        return None

    # Sidebar summary block: only show after results exist
    @render.ui
    def sidebar_results_summary_block():
        df = results_df.get()
        if df.empty:
            return None
        # derive summary
        total_inputs = len(df)
        no_matches = (df.get('status', '').astype(str).str.upper() == 'NO MATCH').sum()
        successful = total_inputs - no_matches
        avg_score = df[df.get('status', '').astype(str).str.upper() != 'NO MATCH']['similarity_score'].mean() if 'similarity_score' in df.columns else None
        avg_score_str = f"{avg_score:.3f}" if avg_score is not None and not pd.isna(avg_score) else "N/A"
        return ui.div(
            ui.h5("Results Summary"),
            ui.p(f"Total Inputs: {total_inputs}"),
            ui.p(f"Successful Matches: {successful}"),
            ui.p(f"No Matches: {no_matches}"),
            ui.p(f"Average Score: {avg_score_str}"),
            class_="alert alert-info alert-animated"
        )
    
    # Run matching process
    @reactive.effect
    @reactive.event(input.run_matching)
    async def run_matching():
        # Disable the button immediately to prevent double-clicks
        ui.update_action_button("run_matching", disabled=True)
        
        # Stay on current tab; navigate to Results after processing
        # Clear existing spinner before processing
        try:
            ui.remove_ui(selector="#processing_spinner")
        except Exception:
            pass
        
        # Get data
        in_df = input_df.get()
        tgt_df = target_df.get()
        
        if in_df.empty or tgt_df.empty:
            ui.notification_show("Please upload both input and target files", type="warning")
            ui.update_action_button("run_matching", disabled=False)  # Re-enable on early return
            return
        
        # Get settings
        in_col = input.input_column()
        tgt_col = input.target_column()
        threshold = input.threshold()
        clean_input_text = input.clean_input()
        clean_target_text = input.clean_target()
        
        if not in_col or not tgt_col:
            ui.modal_show(
                ui.modal(
                    ui.h5("Select Columns First"),
                    ui.p("Please choose the input and target columns to match in the 'Data & Configure' tab before running."),
                    easy_close=True,
                    footer=ui.input_action_button("dismiss_modal", "OK", class_="btn btn-primary")
                )
            )
            ui.update_action_button("run_matching", disabled=False)  # Re-enable on early return
            return
        
        # Get API key for semantic embeddings
        api_key = get_api_key()
        if not api_key:
            ui.notification_show("DeepInfra API key not found. Please set DEEPINFRA_API_KEY", type="error")
            ui.update_action_button("run_matching", disabled=False)  # Re-enable on early return
            return
        
        # Reset circuit for this run
        global _API_FAILURES, FALLBACK_ACTIVE
        _API_FAILURES = 0
        FALLBACK_ACTIVE = False
        
        # Clear previous results
        ui.remove_ui(selector="#process_status > *")
        ui.remove_ui(selector="#process_progress > *")
        ui.remove_ui(selector="#process_summary > *")
        ui.remove_ui(selector="#method_chips > *")
        
        # Add busy indicator at top of page
        ui.busy_indicators.use()
        
        # Use Shiny's built-in Progress with better formatting
        with ui.Progress(min=0, max=100) as p:
            p.set(5, message="Initializing", detail="Preparing data...")
            
            try:
                # Show loading spinner during processing
                try:
                    ui.insert_ui(
                        ui.div(ui.div(class_="spinner"), class_="spinner-container", id="processing_spinner"),
                        selector="#process_progress", where="afterBegin"
                    )
                except Exception:
                    pass
                # Prepare data
                input_list = in_df[in_col].dropna().tolist()
                target_list = tgt_df[tgt_col].dropna().tolist()
                
                # Remove duplicates from target
                target_list_unique = list(dict.fromkeys(target_list))
                
                # Apply cleaning to display text if toggles are on
                # Store both original and cleaned versions
                input_list_display = clean_text_simple(input_list) if clean_input_text else input_list
                
                p.set(10, message="Data Prepared", 
                     detail=f"{len(input_list):,} inputs • {len(target_list_unique):,} targets")
                await asyncio.sleep(0.1)
                
                # Initialize results with potentially cleaned input text for display
                results = pd.DataFrame({
                    'input_description': input_list_display
                })
                
                # Calculate progress steps (semantic only)
                total_methods = 1
                progress_per_method = 80 / total_methods
                current_progress = 10
                
                # Run semantic embeddings only
                effective_batch = min(EMBED_BATCH_SIZE, 1024)
                
                # Check if we'll be using CPU and notify IMMEDIATELY
                if MODEL_FALLBACK_MODE == "local":
                    ui.notification_show(
                        "Using LOCAL CPU for embeddings. This will take significantly longer than API/GPU processing. "
                        f"Processing {len(input_list)} items may take several minutes.",
                        type="warning",
                        duration=10,
                    )
                    FALLBACK_ACTIVE = True
                elif FALLBACK_ACTIVE:  # Already in fallback from previous failures
                    ui.notification_show(
                        "API unavailable. Using LOCAL CPU fallback - processing will be much slower.",
                        type="warning",
                        duration=8,
                    )
                
                # Update progress message if in fallback mode
                progress_msg = "Computing semantic embeddings..."
                if FALLBACK_ACTIVE or MODEL_FALLBACK_MODE == "local":
                    progress_msg = "Computing semantic embeddings (CPU - this will take time)..."
                p.set(current_progress + 5, 
                     message=progress_msg, 
                     detail=f"Processing {len(input_list):,} items")
                await asyncio.sleep(0.1)

                # Simple progress callback for embedding batches
                batches_total = ((len(input_list) + effective_batch - 1) // effective_batch) + \
                                ((len(target_list_unique) + effective_batch - 1) // effective_batch)
                batch_num = [0]

                def progress_callback(msg: str):
                    # Only increment when a batch completes (Embedding batches or Matching)
                    if msg.startswith("Embedding batches:") or msg.startswith("Matching:"):
                        batch_num[0] = min(batch_num[0] + 1, batches_total)
                        progress_pct = current_progress + (batch_num[0] / batches_total) * progress_per_method
                        p.set(
                            int(progress_pct),
                            message="Computing Embeddings",
                            detail=f"Batch {batch_num[0]:,} of {batches_total:,}"
                        )
                
                if USE_ASYNC:
                    embed_results = await run_embed_match_async(
                        input_list,
                        target_list_unique,
                        api_key,
                        progress_callback=progress_callback,
                        clean_input=clean_input_text,
                        clean_target=clean_target_text,
                    )
                else:
                    embed_results = run_embed_match(
                        input_list,
                        target_list_unique,
                        api_key,
                        progress_callback=progress_callback,
                        clean_input=clean_input_text,
                        clean_target=clean_target_text,
                    )
                
                # Apply cleaning to matched target text if toggle is on
                matched_targets = embed_results['match']
                if clean_target_text:
                    # Clean the matched target text for display
                    matched_targets = clean_text_simple(matched_targets)
                
                results['best_match'] = matched_targets
                results['similarity_score'] = embed_results['score']
                # Keep the best match text without decoration for clean exports
                current_progress += progress_per_method
                p.set(int(current_progress), message="Embeddings Complete", detail="Processing results...")
                await asyncio.sleep(0.1)
                
                # Round scores for display
                for col in results.columns:
                    if 'score' in col:
                        results[col] = results[col].round(4)
                
                # Generate score visualization and status indicators
                for col in results.columns:
                    if 'score' in col.lower() or 'similarity' in col.lower():
                        bar_col = f"{col}_bar"
                        results[bar_col] = results[col].apply(create_score_bar)
                
                # Add match status column based on score threshold
                if 'best_match' in results.columns:
                    results.insert(0, 'status', results['similarity_score'].apply(
                        lambda s: 'NO MATCH' if (pd.notna(s) and float(s) < float(threshold)) else 'Match'
                    ))
                
                p.set(95, message="Finalizing", detail="Preparing visualizations...")
                
                # Store results
                results_df.set(results)
                
                # Navigate to results tab automatically
                ui.update_navs("workflow_tabs", selected="Step 2: Results")
                
                # Generate summary statistics
                total_inputs = len(results)
                if 'status' in results.columns:
                    no_matches = (results['status'] == 'NO MATCH').sum()
                    successful_matches = total_inputs - no_matches
                    avg_score = results[results['status'] != 'NO MATCH']['similarity_score'].mean()
                    avg_score_str = f"{avg_score:.3f}" if not pd.isna(avg_score) else "N/A"
                else:
                    no_matches = 0
                    successful_matches = total_inputs
                    avg_score_str = "N/A"
                
                p.set(100, message="Complete", detail="Ready to view results")
                await asyncio.sleep(0.5)
                
                # Hide loading spinner
                try:
                    ui.remove_ui(selector="#processing_spinner")
                except:
                    pass
                
                # Show results summary using notifications
                ui.notification_show(
                    f"Processing complete! Generated {len(results)} results.", 
                    type="success",
                    duration=5
                )
                
                # Insert summary into the page
                # Replace previous summary (if any), then insert a fresh one with animation
                try:
                    ui.remove_ui(selector="#results_summary_msg")
                except Exception:
                    pass
                ui.insert_ui(
                    ui.div(
                        ui.p(f"Total Inputs: {total_inputs}"),
                        ui.p(f"Successful Matches: {successful_matches}"),
                        ui.p(f"No Matches: {no_matches}"),
                        ui.p(f"Average Score: {avg_score_str}"),
                        class_="alert alert-info alert-animated",
                        id="results_summary_msg"
                    ),
                    selector="#sidebar_results_summary",
                    where="afterBegin"
                )
                
                # Show processing method indicator
                # Update method chip to reflect CPU fallback if active
                method_label = "Semantic Embedding (thenlper/gte-large)"
                if FALLBACK_ACTIVE:
                    method_label += " — CPU fallback"
                
                ui.insert_ui(
                    ui.div(
                        ui.span(method_label, class_="method-chip"),
                        class_="text-center mt-3"
                    ),
                    selector="#method_chips",
                    where="afterBegin"
                )
                # After processing completes, navigate to Results tab
                ui.update_navs("workflow_tabs", selected="Step 2: Results")
                
                # Re-enable the button after successful completion
                ui.update_action_button("run_matching", disabled=False)
                
            except Exception as e:
                # Hide loading spinner on error
                try:
                    ui.remove_ui(selector="#processing_spinner")
                except:
                    pass
                ui.notification_show(f"Error: {str(e)}", type="error", duration=10)
                # Re-enable the button on error
                ui.update_action_button("run_matching", disabled=False)
                raise
    
    # Filter and display results
    @render.table
    def results_table():
        df = results_df.get()
        try:
            if input.use_grid():
                return pd.DataFrame()
        except Exception:
            pass
        if df.empty:
            return pd.DataFrame()
        
        # Apply filters
        filtered_df = df.copy()
        
        # Apply search debouncing
        search_term = debounced_search()
        if search_term and search_term.strip():
            mask = filtered_df.apply(
                lambda row: row.astype(str).str.contains(search_term, case=False, na=False).any(),
                axis=1
            )
            filtered_df = filtered_df[mask]
        
        # NO MATCH filter
        if input.show_no_match():
            if 'status' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['status'] == 'NO MATCH']
        
        # Sort by score
        if input.sort_by_score():
            score_cols = [col for col in filtered_df.columns if 'score' in col.lower()]
            if score_cols:
                filtered_df = filtered_df.sort_values(score_cols[0], ascending=False)

        return filtered_df
    
    # Export All Data - includes original columns from input and target CSVs
    @render.download(filename=lambda: f"all_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    def download_all_data():
        df = results_df.get()
        in_df = input_df.get()
        tgt_df = target_df.get()
        
        if df.empty:
            return io.BytesIO(b"No results to export")
        
        # Start with the original input dataframe
        export_df = in_df.copy()
        
        # Check if text cleaning was applied to input column
        # If so, replace the input column with the cleaned version from results
        in_col = input.input_column()
        if input.clean_input() and in_col and 'input_description' in df.columns:
            # Replace the original column with the cleaned version
            export_df[in_col] = df['input_description'].values[:len(export_df)]
        
        # Add the status and similarity score columns from results
        if 'status' in df.columns:
            export_df['status'] = df['status'].values[:len(export_df)]
        if 'similarity_score' in df.columns:
            export_df['similarity_score'] = df['similarity_score'].values[:len(export_df)]
        
        # Add the matched target data (already cleaned if toggle was on)
        if 'best_match' in df.columns:
            export_df['matched_target'] = df['best_match'].values[:len(export_df)]
            
            # Try to merge with target dataframe to get all target columns
            # This assumes the target column selected contains unique identifiers
            tgt_col = input.target_column()
            if tgt_col and not tgt_df.empty:
                # Create a mapping from target description to full target row
                tgt_df_unique = tgt_df.drop_duplicates(subset=[tgt_col])
                # Merge based on the matched values
                merged = export_df.merge(
                    tgt_df_unique,
                    left_on='matched_target',
                    right_on=tgt_col,
                    how='left',
                    suffixes=('', '_target')
                )
                export_df = merged
        
        # Remove UI-only columns like score bars
        bar_cols = [c for c in export_df.columns if c.endswith('_bar')]
        export_df = export_df.drop(columns=bar_cols, errors='ignore')
        
        return io.BytesIO(export_df.to_csv(index=False).encode())
    
    # Export Matches - current functionality (results with mappings)
    @render.download(filename=lambda: f"matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    def download_matches():
        df = results_df.get()
        if not df.empty:
            # Remove UI-only columns like score bars
            export_df = df.copy()
            bar_cols = [c for c in export_df.columns if c.endswith('_bar')]
            export_df = export_df.drop(columns=bar_cols, errors='ignore')
            return io.BytesIO(export_df.to_csv(index=False).encode())
        return io.BytesIO(b"No results to download")

    # Build interactive grid (Tabulator)
    @render.ui
    def results_tabulator():
        df = results_df.get()
        if df.empty:
            return None
        # Convert DataFrame to records for Tabulator
        records = df.to_dict(orient='records')
        cols = []
        for c in df.columns:
            col = {"title": c, "field": c}
            lc = c.lower()
            if ("score" in lc) or ("similarity" in lc):
                col["hozAlign"] = "right"
                col["sorter"] = "number"
            # Disable filter/sort for bar columns
            if lc.endswith("_bar"):
                col["headerFilter"] = False
                col["headerSort"] = False
            else:
                col["headerFilter"] = "input"
            cols.append(col)
        data_json = json.dumps(records)
        cols_json = json.dumps(cols)
        html = (
            "<div style=\"height:600px\" id=\"tabulator_results\"></div>\n"
            "<script>\n"
            "(function(){\n"
            "  function ensureTabulatorAssets(cb){\n"
            "    var cssId='tabulator-css';\n"
            "    if(!document.getElementById(cssId)){\n"
            "      var l=document.createElement('link'); l.id=cssId; l.rel='stylesheet'; l.href='https://unpkg.com/tabulator-tables@5.5.2/dist/css/tabulator.min.css'; document.head.appendChild(l);\n"
            "    }\n"
            "    if(window.Tabulator){ cb(); return; }\n"
            "    var s=document.createElement('script'); s.src='https://unpkg.com/tabulator-tables@5.5.2/dist/js/tabulator.min.js'; s.onload=cb; document.body.appendChild(s);\n"
            "  }\n"
            "  function init(){\n"
            "    var el = document.getElementById('tabulator_results');\n"
            "    if(!el) return;\n"
            "    if (el._tabulator) { el._tabulator.destroy(); }\n"
            "    var table = new Tabulator(el, {\n"
            "      data: " + data_json + ",\n"
            "      reactiveData: false,\n"
            "      layout: 'fitDataStretch',\n"
            "      height: '600px',\n"
            "      movableColumns: true,\n"
            "      resizableColumns: true,\n"
            "      columnDefaults: {headerSort: true, editable: false},\n"
            "      columns: " + cols_json + ",\n"
            "      initialSort: [{column: 'similarity_score', dir: 'desc'}],\n"
            "      rowFormatter: function(row){\n"
            "        var data = row.getData();\n"
            "        if(data.status === 'NO MATCH'){\n"
            "          row.getElement().style.backgroundColor = 'rgba(220, 53, 69, 0.08)';\n"
            "        }\n"
            "      }\n"
            "    });\n"
            "    el._tabulator = table;\n"
            "  }\n"
            "  ensureTabulatorAssets(init);\n"
            "})();\n"
            "</script>\n"
        )
        return ui.HTML(html)

    # Chart description helper
    @render.ui
    def chart_description():
        viz_type = input.plotly_viz_type()
        descriptions = {
            "density": "**Density Plot**: Shows probability density of scores. Higher peaks = more concentrated scores. Rug plot below shows individual points.",
            "histogram": "**Histogram**: Frequency of scores in bins. Height = count in each range. Shows mean and median.",
            "threshold": "**Threshold Analysis**: Shows how match percentage changes at different threshold values. Helps find optimal cutoff point."
            # HIDDEN DESCRIPTIONS - Keep in sync with hidden visualizations above
            # "box": "**Box Plot**: Shows quartiles (Q1, median, Q3) and outliers. Box = middle 50% of data.",
            # "violin": "**Violin Plot**: Distribution shape split by match/no-match status. Width = data density.",
            # "scatter": "**Scatter Plot**: Each point is one item (index vs score). Blue = match, red = no match. Shows sequential patterns.",
            # "ecdf": "**Cumulative Distribution**: Shows % of data at or below each score. Answers 'what % scores below X?'",
            # "sunburst": "**Match Breakdown**: Hierarchical view of match statistics. Inner ring shows overall split, outer rings show score ranges."
        }
        desc = descriptions.get(viz_type, "")
        if desc:
            return ui.div(
                ui.markdown(desc),
                class_="border rounded",
                style="padding: 0.75rem 1rem; margin-bottom: 1rem; font-size: 0.9rem; background-color: var(--bs-tertiary-bg, transparent); color: inherit;"
            )
        return None
    
    # Interactive Plotly visualizations
    @render_widget
    def plotly_viz():
        # Access reactive values to establish dependencies
        df = results_df.get()
        
        if df.empty:
            # Return empty figure when no data
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_annotation(
                text="Run mapping to view interactive charts",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="#666")
            )
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=400
            )
            return fig
        
        # Check for similarity score column
        if "similarity_score" not in df.columns:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_annotation(
                text="No similarity scores available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="#666")
            )
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=400
            )
            return fig
        
        # Get input values - these trigger reactive updates
        viz_type = input.plotly_viz_type()
        show_threshold = input.show_threshold_line()
        threshold = input.threshold()
        
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np
        
        # Keep full dataframe for scatter plot, filter for other plots
        df_clean = df.dropna(subset=['similarity_score']).copy()
        scores = df_clean["similarity_score"]
        
        # Create figure based on visualization type
        if viz_type == "density":
            # Kernel Density Estimation
            from scipy import stats
            density = stats.gaussian_kde(scores)
            x_range = np.linspace(0, 1, 200)
            y_density = density(x_range)
            
            fig = go.Figure()
            
            # Add density trace
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_density,
                mode='lines',
                fill='tozeroy',
                name='Density',
                line=dict(color='#4e79a7', width=2),
                fillcolor='rgba(78, 121, 167, 0.3)',
                hovertemplate='Score: %{x:.3f}<br>Density: %{y:.3f}<extra></extra>'
            ))
            
            # Add rug plot for actual data points
            fig.add_trace(go.Scatter(
                x=scores,
                y=[-0.01 * max(y_density)] * len(scores),
                mode='markers',
                name='Data points',
                marker=dict(color='#4e79a7', size=2, symbol='line-ns', line=dict(width=1, color='#4e79a7')),
                hovertemplate='Score: %{x:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Similarity Score Density Distribution",
                xaxis_title="Similarity Score",
                yaxis_title="Density",
                showlegend=True
            )
            
        elif viz_type == "histogram":
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=scores,
                nbinsx=30,
                name='Scores',
                marker_color='#4e79a7',
                opacity=0.8,
                hovertemplate='Score range: %{x}<br>Count: %{y}<extra></extra>'
            ))
            
            # Add statistics annotation
            mean_score = scores.mean()
            median_score = scores.median()
            fig.add_annotation(
                text=f"Mean: {mean_score:.3f}<br>Median: {median_score:.3f}",
                xref="paper", yref="paper",
                x=0.98, y=0.98,
                showarrow=False,
                bgcolor="white",
                bordercolor="#4e79a7",
                borderwidth=1
            )
            
            fig.update_layout(
                title="Interactive Histogram of Similarity Scores",
                xaxis_title="Similarity Score",
                yaxis_title="Count",
                bargap=0.05
            )
            
        elif viz_type == "box":
            # Box plot with individual points
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=scores,
                name='Scores',
                marker_color='#4e79a7',
                boxpoints='outliers',
                jitter=0.3,
                pointpos=-1.8,
                hovertemplate='Score: %{y:.3f}<extra></extra>'
            ))
            
            # Add violin for comparison
            fig.add_trace(go.Violin(
                y=scores,
                name='Distribution',
                side='positive',
                opacity=0.3,
                marker_color='#e15759',
                hovertemplate='Score: %{y:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Box Plot with Distribution Shape",
                yaxis_title="Similarity Score",
                showlegend=True
            )
            
        elif viz_type == "violin":
            fig = go.Figure()
            
            # Split by match status using cleaned dataframe
            matched_scores = df_clean[df_clean['status'] != 'NO MATCH']['similarity_score']
            no_match_scores = df_clean[df_clean['status'] == 'NO MATCH']['similarity_score']
            
            if len(matched_scores) > 0:
                fig.add_trace(go.Violin(
                    y=matched_scores,
                    name='Matched',
                    side='negative',
                    marker_color='#4e79a7',
                    line_color='#4e79a7',
                    meanline_visible=True,
                    hovertemplate='Matched<br>Score: %{y:.3f}<extra></extra>'
                ))
            
            if len(no_match_scores) > 0:
                fig.add_trace(go.Violin(
                    y=no_match_scores,
                    name='No Match',
                    side='positive',
                    marker_color='#e15759',
                    line_color='#e15759',
                    meanline_visible=True,
                    hovertemplate='No Match<br>Score: %{y:.3f}<extra></extra>'
                ))
            
            fig.update_layout(
                title="Violin Plot: Score Distribution by Match Status",
                yaxis_title="Similarity Score",
                violingap=0.3,
                violinmode='overlay'
            )
            
        elif viz_type == "scatter":
            # Scatter plot with color by match status
            colors = ['#4e79a7' if status != 'NO MATCH' else '#e15759' 
                     for status in df_clean['status']]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(df_clean))),
                y=df_clean['similarity_score'],
                mode='markers',
                marker=dict(
                    color=colors,
                    size=8,
                    opacity=0.6,
                    line=dict(width=1, color='white')
                ),
                text=df_clean['input_description'],
                hovertemplate='Index: %{x}<br>Score: %{y:.3f}<br>Input: %{text}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Similarity Scores by Index",
                xaxis_title="Item Index",
                yaxis_title="Similarity Score"
            )
            
        elif viz_type == "ecdf":
            # Empirical Cumulative Distribution Function
            sorted_scores = np.sort(scores)
            ecdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sorted_scores,
                y=ecdf,
                mode='lines',
                name='ECDF',
                line=dict(color='#4e79a7', width=2),
                hovertemplate='Score: %{x:.3f}<br>Cumulative %: %{y:.1%}<extra></extra>'
            ))
            
            # Add markers at quartiles
            q25, q50, q75 = np.percentile(scores, [25, 50, 75])
            fig.add_trace(go.Scatter(
                x=[q25, q50, q75],
                y=[0.25, 0.50, 0.75],
                mode='markers+text',
                name='Quartiles',
                marker=dict(color='#e15759', size=10),
                text=['Q1', 'Median', 'Q3'],
                textposition='top center',
                hovertemplate='%{text}<br>Score: %{x:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Empirical Cumulative Distribution",
                xaxis_title="Similarity Score",
                yaxis_title="Cumulative Probability",
                yaxis=dict(tickformat='.0%')
            )
        
        elif viz_type == "threshold":
            # Threshold Analysis - shows match rate at different thresholds
            thresholds = np.linspace(0, 1, 101)
            match_rates = [(scores >= t).mean() for t in thresholds]
            
            fig = go.Figure()
            
            # Main threshold curve
            fig.add_trace(go.Scatter(
                x=thresholds,
                y=match_rates,
                mode='lines',
                name='Match Rate',
                line=dict(color='#4e79a7', width=3),
                hovertemplate='Threshold: %{x:.3f}<br>Match Rate: %{y:.1%}<extra></extra>'
            ))
            
            # Add current threshold marker
            if threshold:
                current_match_rate = (scores >= threshold).mean()
                fig.add_trace(go.Scatter(
                    x=[threshold],
                    y=[current_match_rate],
                    mode='markers+text',
                    name='Current Threshold',
                    marker=dict(color='#e15759', size=12),
                    text=[f'{current_match_rate:.1%}'],
                    textposition='top center',
                    hovertemplate='Current: %{x:.3f}<br>Rate: %{y:.1%}<extra></extra>'
                ))
                
                # Add reference lines
                fig.add_vline(x=threshold, line_dash="dash", line_color="#e15759", opacity=0.5)
                fig.add_hline(y=current_match_rate, line_dash="dot", line_color="#e15759", opacity=0.5)
            
            # Add 50% reference line
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.3)
            
            fig.update_layout(
                title="Threshold Analysis - Match Rate vs Cutoff",
                xaxis_title="Threshold Value",
                yaxis_title="Match Rate",
                yaxis=dict(tickformat='.0%', range=[0, 1.05]),
                xaxis=dict(range=[0, 1])
            )
        
        # Hidden chart - uncomment block to re-enable
        # elif viz_type == "sunburst":
        #     # Sunburst chart - hierarchical match breakdown
        #     # Create hierarchical data for sunburst
        #     total = len(df_clean)
        #     
        #     # Calculate match/no-match counts
        #     if 'status' in df_clean.columns:
        #         match_mask = df_clean['status'] != 'NO MATCH'
        #     else:
        #         match_mask = df_clean['similarity_score'] >= threshold if threshold else df_clean['similarity_score'] >= 0.85
        #     
        #     matched = match_mask.sum()
        #     no_match = total - matched
        #     
        #     # Create score bins for matched items
        #     matched_df = df_clean[match_mask]
        #     
        #     # Define score ranges
        #     labels = []
        #     parents = []
        #     values = []
        #     colors = []
        #     
        #     # Root level
        #     labels.append("All Items")
        #     parents.append("")
        #     values.append(total)
        #     colors.append("#94a3b8")
        #     
        #     # Match/No Match level
        #     if matched > 0:
        #         labels.append("Matched")
        #         parents.append("All Items")
        #         values.append(matched)
        #         colors.append("#4e79a7")
        #         
        #         # Score ranges for matched items - dynamic based on threshold
        #         if len(matched_df) > 0:
        #             scores_matched = matched_df['similarity_score']
        #             
        #             # Calculate dynamic ranges based on user's threshold
        #             # Use the actual threshold value, defaulting to 0.85 if not set
        #             thresh_val = threshold if threshold else 0.85
        #             
        #             # High confidence: threshold + 0.10 or higher (capped at 1.0)
        #             high_threshold = min(thresh_val + 0.10, 1.0)
        #             high_conf = (scores_matched >= high_threshold).sum()
        #             if high_conf > 0:
        #                 labels.append(f"High (≥{high_threshold:.2f}): {high_conf}")
        #                 parents.append("Matched")
        #                 values.append(high_conf)
        #                 colors.append("#059669")
        #             
        #             # Good confidence: threshold + 0.05 to threshold + 0.10
        #             good_lower = thresh_val + 0.05
        #             good_upper = high_threshold
        #             if good_lower < 1.0:  # Only show if range is valid
        #                 good_conf = ((scores_matched >= good_lower) & (scores_matched < good_upper)).sum()
        #                 if good_conf > 0:
        #                     labels.append(f"Good ({good_lower:.2f}-{good_upper:.2f}): {good_conf}")
        #                     parents.append("Matched")
        #                     values.append(good_conf)
        #                     colors.append("#0ea5e9")
        #             
        #             # Moderate confidence: threshold to threshold + 0.05
        #             mod_lower = thresh_val
        #             mod_upper = min(thresh_val + 0.05, 1.0)
        #             moderate_conf = ((scores_matched >= mod_lower) & (scores_matched < mod_upper)).sum()
        #             if moderate_conf > 0:
        #                 labels.append(f"Moderate ({mod_lower:.2f}-{mod_upper:.2f}): {moderate_conf}")
        #                 parents.append("Matched")
        #                 values.append(moderate_conf)
        #                 colors.append("#8b5cf6")
        #             
        #             # Note: There shouldn't be any "Low" matches below threshold
        #             # since we filter by threshold, but check just in case
        #             low_conf = (scores_matched < thresh_val).sum()
        #             if low_conf > 0:
        #                 labels.append(f"Below threshold (<{thresh_val:.2f}): {low_conf}")
        #                 parents.append("Matched")
        #                 values.append(low_conf)
        #                 colors.append("#f59e0b")
        #     
        #     if no_match > 0:
        #         labels.append("No Match")
        #         parents.append("All Items")
        #         values.append(no_match)
        #         colors.append("#e15759")
        #     
        #     # Create sunburst
        #     fig = go.Figure(go.Sunburst(
        #         labels=labels,
        #         parents=parents,
        #         values=values,
        #         branchvalues="total",
        #         marker=dict(colors=colors),
        #         textinfo="label+percent parent",
        #         hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percentParent}<extra></extra>'
        #     ))
        #     
        #     fig.update_layout(
        #         title="Match Statistics Breakdown",
        #         height=500
        #     )
        
        # Add threshold line if requested (but not for sunburst chart where it doesn't apply)
        if show_threshold and threshold and viz_type != "sunburst":
            fig.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold: {threshold:.2f}",
                annotation_position="top"
            )
        
        # Common layout updates (individual charts already set their axis ranges)
        fig.update_layout(
            template="plotly_white",
            hovermode='closest',
            height=500,
            margin=dict(l=50, r=50, t=50, b=50),
            font=dict(family="system-ui, -apple-system, sans-serif")
        )
        
        # Return the Plotly figure directly for render_widget
        return fig

# Create the app
app = App(app_ui, server)

