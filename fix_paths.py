import sqlite3, os

APP_ROOT = "C:/Users/vinay/SIH-2026-DR-Screening/app"
DB = os.path.join(APP_ROOT, "database", "dr_screening.db")

conn = sqlite3.connect(DB)
cur = conn.cursor()

for row in cur.execute("SELECT screening_id, image_path, gapcam_path FROM screenings").fetchall():
    sid, img, gap = row

    def normalize(path):
        if not path:
            return None
        path = path.strip().replace("\\", "/")
        # If absolute Windows path starting with C: or similar
        if len(path) > 2 and path[1] == ":":
            # Strip app root prefix if present
            app_prefix = APP_ROOT.replace("\\", "/")
            if path.startswith(app_prefix):
                rel = path[len(app_prefix):].lstrip("/")
            else:
                # Extract just the uploads/... part
                rel = "/".join(path.split("/")[-3:])  # uploads/filename
            return rel
        # Already relative — ensure it starts with uploads/
        if not path.startswith("uploads/"):
            path = "uploads/" + path.lstrip("/")
        return path

    new_img = normalize(img)
    new_gap = normalize(gap)

    print(f"{sid}: img={new_img}, gap={new_gap}")
    cur.execute("UPDATE screenings SET image_path=?, gapcam_path=? WHERE screening_id=?",
                (new_img, new_gap, sid))

conn.commit()
conn.close()
print("Done.")
