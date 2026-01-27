# Deploying AI - Learning Support Cheat Sheet

## 📌 Module Overview
**Topic**: Design of AI Systems (based on Chip Huyen's "AI Engineering").
**Goal**: From ML systems to Foundation Models, RAG, Agents, and Optimization.
**Schedule**: 9 Live Sessions across ~3 weeks.
**Primary Contact**: Vishnou Vinayagame (Learning Support).

## 📂 Repository Structure

### `01_materials` (Course Content)
*   **`slides/`**: 9 PDF decks corresponding to the sessions.
    *   *Note*: The file numbering (`01_introduction.pdf` ... `09_optimization.pdf`) maps cleanly to the schedule in `README.md`.
*   **`labs/`**: Interactive Jupyter notebooks.
    *   These contain the code demos.
    *   **Mapping**:
        *   `01_*.ipynb` -> Session 1 (Intro)
        *   `02_*.ipynb` -> Session 2 (Foundation Models/Embeddings)
        *   `03_*.ipynb` -> Session 3 (Evaluation)
        *   ...and so on.
    *   *Tip*: Point students here for "how-to" code examples.

### `02_activities` (Assessment)
*   **`assignment_1.ipynb`**: **Due Nov 3 (@3pm)**. Topic likely: Embeddings/Search (based on file placement/timing).
*   **`assignment_2.md`**: **Due Nov 10 (@3pm)**. Final assignment.
*   *Note*: Students submit these via Pull Request.

### `03_instructional_team` (Internal)
*   contains `generate_slides.sh` (for building slides).
*   **`README.md` (Playbook)**: ⚠️ **Caution**. The content descriptions in this specific file ("Week 1: Data Eng") appear **outdated** compared to the root `README.md`. Reliability is low for topic descriptions, but high for **process** (grading via PRs).

### `04_this_cohort`
*   Place for uploading live coding files created during sessions.

## 🗓️ Schedule & Assessment
**Grading**: Pass/Fail (Min 60 points).
*   **Quizzes (60%)**: 6 Quizzes (Administered during live sessions).
*   **Assignment 1 (20%)**: Complete = 100pts, Incomplete = 50pts.
*   **Assignment 2 (20%)**: Complete = 100pts, Incomplete = 50pts.

**Live Sessions**:
1.  Intro to AI Systems
2.  Foundation Models
3.  Evaluations
4.  Prompt Engineering
5.  RAG
6.  Agents
7.  Finetuning
8.  Data Engineering
9.  Optimization

## 🛠️ Environment Setup (Support FAQ)
Students use **`uv`** for Python management.

**Common Fix for "Package not found":**
1.  Ensure `uv` is installed.
2.  Run setup:
    ```bash
    uv venv deploying-ai-env --python 3.11
    source deploying-ai-env/bin/activate  # (or Scripts/activate on Windows)
    uv sync --active
    ```
3.  **VS Code**: Ensure they select the `deploying-ai-env` kernel for notebooks.

## 📝 Staff Workflow
1.  **Work Periods**: Be available on Zoom. Answer questions.
2.  **Grading**:
    *   Check Pull Requests.
    *   **Complete**: Code works, no errors.
    *   **Incomplete**: Buggy or broken.
    *   **Action**: Tag yourself on their PR, review, and leave constructive feedback.
