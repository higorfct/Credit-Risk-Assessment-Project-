from src.main import main
import os
import joblib

def test_pipeline_integration(tmp_path):
    # Configura o path temporário para salvar artefatos
    os.environ["ARTIFACTS_PATH"] = str(tmp_path)

    # Executa o pipeline completo
    main()

    # Verifica se os artefatos foram criados
    assert os.path.exists(tmp_path / "RandomForest_model.joblib")
    assert os.path.exists(tmp_path / "scaler.joblib")
    assert os.path.exists(tmp_path / "selector.joblib")
