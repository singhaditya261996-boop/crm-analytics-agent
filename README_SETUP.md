# CRM Analytics Agent — Setup Guide

This guide gets the tool running on your computer.
You only need to do this once. After that, starting takes about 10 seconds.

---

## Before You Start

Run a quick check to make sure your computer is ready:

**Mac:**
1. Open Terminal (press Cmd + Space, type Terminal, press Enter)
2. Drag the file "check_requirements.sh" into the Terminal window
3. Press Enter
4. Follow any fix instructions it gives you

**Windows:**
1. Open Command Prompt (press Windows key, type cmd, press Enter)
2. Type: `cd Desktop\crm-agent`  (adjust path if saved elsewhere)
3. Type: `bash check_requirements.sh`
4. Follow any fix instructions it gives you

---

## Step 1 — Install Docker Desktop
Time needed: 5 minutes

1. Go to: docker.com/products/docker-desktop
2. Click Download for Mac or Download for Windows
3. Install it like any normal app (drag to Applications on Mac,
   run the installer on Windows)
4. Open Docker Desktop
5. Wait until the bottom left shows a green dot and says "Running"

**Windows users — if Docker asks about WSL2:**
Say yes to everything. If you get an error, open Command Prompt
as Administrator and run:
```
wsl --install
```
Then restart your computer and open Docker Desktop again.

---

## Step 2 — Get the Project Folder

1. Download the zip file your team lead shared
2. Unzip it (double-click on Mac, right-click → Extract All on Windows)
3. You should see a folder called "crm-agent"
4. Put it somewhere easy to find — your Desktop is fine

Do NOT rename the folder.

---

## Step 3 — First Launch
The first launch downloads the AI model (about 9GB total).
This takes 10-30 minutes depending on your internet speed.
After this, every launch takes about 10 seconds.

**Mac:**
1. Open Terminal
2. Type: `cd Desktop/crm-agent`  (adjust if saved elsewhere)
3. Type: `docker-compose up`
4. Watch the text — this is all normal
5. When you see "You can now view your Streamlit app":
   Open your browser and go to: **http://localhost:8501**

**Windows:**
1. Open Command Prompt
2. Type: `cd Desktop\crm-agent`  (adjust if saved elsewhere)
3. Type: `docker-compose up`
4. Watch the text — this is all normal
5. When you see "You can now view your Streamlit app":
   Open your browser and go to: **http://localhost:8501**

You should see the CRM Analytics Agent dashboard.

---

## Step 4 — Adding Your Data Files

1. Find the "uploads" folder inside the crm-agent folder
   (crm-agent → data → uploads)
2. Drop your Excel or CSV files in there
3. Go back to your browser — the files appear automatically

The tool accepts: .xlsx, .xls, .csv, .pptx, .pdf, .png, .jpg

---

## Step 5 — Every Launch After the First One

1. Make sure Docker Desktop is open (green dot in bottom left)
2. Open Terminal or Command Prompt
3. Navigate to the crm-agent folder (`cd Desktop/crm-agent`)
4. Type: `docker-compose up`
5. Open browser: **http://localhost:8501**

Or if you prefer a shortcut, type: `make start`

---

## Step 6 — Stopping the Tool

In the terminal window where the tool is running:
Press **Ctrl + C**

Or type in a new terminal window:
```
docker-compose down
```

---

## Troubleshooting

**Problem: "Port 8501 is already in use"**
```
Mac/Linux: lsof -ti:8501 | xargs kill -9
Windows:   netstat -ano | findstr :8501
           then: taskkill /PID [the number shown] /F
```

**Problem: "The download seems stuck or very slow"**
This is normal — 9GB takes time on slow connections.
You can check progress in the terminal — it shows a percentage.
Leave it running and come back later.

**Problem: "Answers are very slow"**
Your computer may have limited RAM.
Ask your team lead to run: `make switch-small`
This switches to a faster, lighter AI model.

**Problem: "I can't find the uploads folder"**
```
Mac:     Open Finder → navigate to crm-agent → data → uploads
Windows: Open File Explorer → navigate to crm-agent → data → uploads
```

**Problem: "Docker Desktop won't start on Windows"**
Open Command Prompt as Administrator and run:
```
wsl --update
```
Then restart your computer and try Docker Desktop again.

**Problem: "Error says 'cannot connect to Docker daemon'"**
Docker Desktop isn't running.
Open Docker Desktop and wait for the green dot before trying again.

**Still stuck?**
Take a screenshot of the error message and send it to your team lead.
