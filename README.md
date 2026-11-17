# DiploMind

Adaptive AI-powered International Relations literacy platform for Pakistani learners (FYP F26-172).

- **Backend:** FastAPI (Python), runs in Docker
- **Database:** MongoDB, runs in Docker
- **Frontend:** React + Vite + Tailwind, runs with `npm run dev`

---

## Folder structure

```
DiploMind/
├─ docker-compose.yml      Starts MongoDB + backend containers
├─ .env.example            Template for .env (copy it, never commit .env)
├─ .gitignore              Files Git should ignore (.env, node_modules, ...)
│
├─ backend/                FastAPI server
│  ├─ Dockerfile           How the backend container is built
│  ├─ requirements.txt     Python packages for the backend
│  └─ app/
│     ├─ main.py           Creates the app, registers routers, /api/health
│     ├─ db.py             MongoDB connection, collections and indexes
│     └─ auth.py           Signup, login, /me, and current_user (login check)
│
├─ frontend/               React app (the UI screens)
│  ├─ vite.config.ts       Dev server; sends /api requests to the backend
│  ├─ package.json         Frontend packages
│  └─ src/
│     ├─ App.tsx           Routes: which URL shows which page
│     ├─ api.ts            All calls from the UI to the backend
│     ├─ components/       Shared pieces (e.g. Header)
│     └─ pages/            One file per screen (Login, Signup, Onboarding, Chat...)
│
└─ data_scrapper/          Scrapers that build the article knowledge base
```

## Databases (inside the one MongoDB container)

| Database    | Collection | What's in it                        | Written by          |
|-------------|------------|-------------------------------------|---------------------|
| `diplomind` | `users`    | Accounts and profile (register, depth, onboarding) | backend (`auth.py`) |
| `irlit`     | `articles` | Scraped articles                    | `data_scrapper/`    |

View them in MongoDB Compass at `mongodb://localhost:27017`.

**Note:** each laptop has its own database. GitHub shares the code, not the data.

---

## First-time setup (each team member)

1. Install Docker Desktop, Git, Node.js LTS (and optionally MongoDB Compass).
2. In PowerShell (once): `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
3. Clone the repo, then in the `DiploMind` folder:
   ```powershell
   Copy-Item .env.example .env      # then put a long random string in JWT_SECRET
   docker compose up --build -d
   cd frontend
   npm install
   npm run dev
   ```
4. Open http://localhost:5173/signup

## Daily routine

```powershell
git pull
docker compose up -d          # add --build if requirements.txt or Dockerfile changed
cd frontend
npm run dev                   # run npm install first if package.json changed
```

After making changes:
```powershell
git add .
git commit -m "what you changed"
git pull
git push
```

Stop everything with `docker compose down`. **Never use `down -v`**, because it deletes the database.

Useful links: API docs at http://localhost:8000/docs · backend logs with `docker compose logs backend -f`

---

## Where to make changes

### Adding a new backend module (e.g. onboarding quiz, chat)
1. Create a new file in `backend/app/`, e.g. `onboarding.py`, with its own `router = APIRouter(prefix="/api/onboarding")`.
2. Register it in `backend/app/main.py`: `app.include_router(onboarding_router)`.
3. New collection? Add it in `backend/app/db.py` (e.g. `questions = db["onboarding_questions"]`), and any index in `init_indexes()`.
4. Route needs a logged-in user? Add `user: dict = Depends(current_user)` (import it from `.auth`).
5. New Python package? Add it to `backend/requirements.txt`, then run `docker compose up --build -d`.

All backend URLs start with `/api/...`. Python file edits reload automatically, so no restart is needed.

### Adding a new screen
1. Create the page in `frontend/src/pages/`, e.g. `ProfilePage.tsx`.
2. Add its route in `frontend/src/App.tsx`: `<Route path="/profile" element={<ProfilePage />} />`.
3. Shared UI pieces go in `frontend/src/components/`.

### Connecting a screen to the backend
- Add a function for the new endpoint in `frontend/src/api.ts`, not inside the page, so every API call stays in one place.
- For routes that need login, send the saved token in a header:
  `Authorization: Bearer ${localStorage.getItem("token")}`

### Adding a setting or secret (e.g. an API key)
- Add it to `.env` (your own value) **and** to `.env.example` (placeholder only), so teammates know it exists.
- Read it in Python with `os.environ["NAME"]`.

### Rules
- Never commit `.env` or `node_modules`.
- Pull before you push.
- Keep the report's data dictionary (Section 4.9) in sync when you add or rename collection fields.