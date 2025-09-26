from fastapi import FastAPI
from app.api.endpoints import router as api_router
from app.api.unified_endpoints import router as unified_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Loan Approval API",
    description="API para predicción de aprobación de préstamos",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir rutas
app.include_router(api_router, prefix="/api/v1")
app.include_router(unified_router)
@app.get("/")
async def root():
    return {"message": "Loan Approval API - Documentación en /docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)