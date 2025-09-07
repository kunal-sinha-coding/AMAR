import zipfile
import os
import tempfile
import subprocess
import shutil
import time

# === CONFIG ===
db_zip_path = "virtuoso_db.zip"        # path to your zip
virtuosoPath = "../virtuoso-opensource"  # path to Virtuoso binaries
port = 3001

# === STEP 1: Create temporary folder for DB extraction ===
tmpdir = tempfile.mkdtemp(prefix="virtuoso_")
print(f"Temporary folder: {tmpdir}")

try:
    # === STEP 2: Open zip and extract only required DB files ===
    with zipfile.ZipFile(db_zip_path, 'r') as z:
        # Filter out files inside virtuoso_db/ (skip directories)
        db_files = [f for f in z.namelist() if f.startswith("virtuoso_db/") and not f.endswith("/")]
        
        for file_name in db_files:
            print(f"Extracting {file_name} ...")
            z.extract(file_name, path=tmpdir)

    # Full path to the main DB folder
    db_folder = os.path.join(tmpdir, "virtuoso_db")
    db_file_path = os.path.join(db_folder, "virtuoso.db")

    if not os.path.exists(db_file_path):
        raise FileNotFoundError(f"DB file not found at {db_file_path}")

    # === STEP 3: Create temporary ini file pointing to extracted DB ===
    ini_file_path = os.path.join(tmpdir, "virtuoso.ini")
    with open(ini_file_path, "w") as f:
        f.write(f"""
[Database]
DatabaseFile={db_file_path}
NumberOfBuffers=500000
MaxDirtyBuffers=250000

[HTTPServer]
ServerPort={port}
""")

    # === STEP 4: Start Virtuoso server ===
    print("Starting Virtuoso server...")
    proc = subprocess.Popen(
        [f"{virtuosoPath}/bin/virtuoso-t", f"+configfile={ini_file_path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait a few seconds for startup
    time.sleep(5)

    # Optional: check if server started successfully
    if proc.poll() is not None:
        stdout, stderr = proc.communicate()
        print(stdout.decode())
        print(stderr.decode())
        raise RuntimeError("Virtuoso server failed to start")

    print("Virtuoso server is running. Press Ctrl+C to stop.")

    # Keep the script running so server stays alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping Virtuoso server...")

    proc.terminate()
    proc.wait()

finally:
    # === STEP 5: Clean up temporary folder ===
    print(f"Cleaning up temporary folder {tmpdir} ...")
    shutil.rmtree(tmpdir)
