import argparse

import psycopg2

from lib.settings import database_config

def init_schema(*, reset: bool = False) -> None:
    """Create or update the TimescaleDB schema.

    Existing tables are retained unless ``reset`` is explicitly enabled.
    """
    
    conn = psycopg2.connect(**database_config())
    conn.autocommit = True 
    cursor = conn.cursor()

    try:
        print("Enabling TimescaleDB extension...")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")

        if reset:
            print("Reset requested; dropping existing DASH tables...")
            cursor.execute("DROP TABLE IF EXISTS processing_logs CASCADE;")
            cursor.execute("DROP TABLE IF EXISTS bls_time_offset CASCADE;")
            cursor.execute("DROP TABLE IF EXISTS bls_tle_2param CASCADE;")
            cursor.execute("DROP TABLE IF EXISTS bls_orbit_determination CASCADE;")
            cursor.execute("DROP TABLE IF EXISTS spacecraft_registry CASCADE;")

        # ---------------------------------------------------------
        # TABLE 1: Spacecraft Registry
        # ---------------------------------------------------------
        print("Creating spacecraft_registry table...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS spacecraft_registry (
            spacecraft_id VARCHAR(50) PRIMARY KEY,
            name VARCHAR(100),
            center_frequency_hz FLOAT NOT NULL,
            track BOOL,
            last_updated TIMESTAMPTZ DEFAULT NOW(),
            frequency_confidence VARCHAR(20) DEFAULT 'UNVERIFIED'
        );
        """)

        # ---------------------------------------------------------
        # TABLE 2: BLS Time Offset (Batch)
        # ---------------------------------------------------------
        print("Creating bls_time_offset table...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS bls_time_offset (
            timestamp TIMESTAMPTZ NOT NULL,
            spacecraft_id VARCHAR(50) NOT NULL REFERENCES spacecraft_registry(spacecraft_id),
            name VARCHAR(100),
            contact_id VARCHAR(500),
            model_type VARCHAR(20) NOT NULL,
            
            time_shift_s FLOAT,
            doppler_bias_hz FLOAT,
            fitted_frequency_hz FLOAT,
            freq_error_hz FLOAT,
            
            var_time_shift FLOAT,
            var_doppler_bias FLOAT,
            var_freq_error FLOAT,
            
            ssr FLOAT,
            aic FLOAT,
            bic FLOAT,
            chi_square_reduced FLOAT,
            
            -- Residual Analysis and Noise Metrics
            num_samples INT,
            residual_rmse FLOAT,
            residual_mean FLOAT,
            residual_acf_lag1 FLOAT,
            residual_pacf_lag1 FLOAT,
            residual_significance_threshold FLOAT,
            is_white_noise BOOL,
            
            metadata_json JSONB,             
            
            PRIMARY KEY (timestamp, spacecraft_id, model_type)
        );
        """)
        cursor.execute("SELECT create_hypertable('bls_time_offset', 'timestamp', if_not_exists => TRUE);")

        # ---------------------------------------------------------
        # TABLE 3: BLS TLE Update (2-Parameter Fit)
        # ---------------------------------------------------------
        print("Creating bls_tle_2param table...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS bls_tle_2param (
            timestamp TIMESTAMPTZ NOT NULL,
            spacecraft_id VARCHAR(50) NOT NULL REFERENCES spacecraft_registry(spacecraft_id),
            name VARCHAR(100),
            contact_id VARCHAR(500),
            model_type VARCHAR(20) NOT NULL,
            
            mean_anomaly FLOAT,
            mean_motion FLOAT,
            doppler_bias_hz FLOAT,
            
            var_mean_anomaly FLOAT,
            var_mean_motion FLOAT,
            var_doppler_bias FLOAT,
            
            ssr FLOAT,
            aic FLOAT,
            bic FLOAT,
            chi_square_reduced FLOAT,
            num_samples INT,
            residual_rmse FLOAT,
            residual_mean FLOAT,
            residual_acf_lag1 FLOAT,
            residual_pacf_lag1 FLOAT,
            residual_significance_threshold FLOAT,
            is_white_noise BOOL,
            updated_tle_line1 TEXT,
            updated_tle_line2 TEXT,
            metadata_json JSONB,             
            
            PRIMARY KEY (timestamp, spacecraft_id, model_type)
        );
        """)
        cursor.execute("""
        ALTER TABLE bls_tle_2param
            ADD COLUMN IF NOT EXISTS num_samples INT,
            ADD COLUMN IF NOT EXISTS residual_rmse FLOAT,
            ADD COLUMN IF NOT EXISTS residual_mean FLOAT,
            ADD COLUMN IF NOT EXISTS residual_acf_lag1 FLOAT,
            ADD COLUMN IF NOT EXISTS residual_pacf_lag1 FLOAT,
            ADD COLUMN IF NOT EXISTS residual_significance_threshold FLOAT,
            ADD COLUMN IF NOT EXISTS is_white_noise BOOL,
            ADD COLUMN IF NOT EXISTS updated_tle_line1 TEXT,
            ADD COLUMN IF NOT EXISTS updated_tle_line2 TEXT;
        """)
        cursor.execute("SELECT create_hypertable('bls_tle_2param', 'timestamp', if_not_exists => TRUE);")

        # ---------------------------------------------------------
        # TABLE 4: BLS Orbit Determination (Batch IOD)
        # ---------------------------------------------------------
        print("Creating bls_orbit_determination table...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS bls_orbit_determination (
            timestamp TIMESTAMPTZ NOT NULL,
            spacecraft_id VARCHAR(50) NOT NULL REFERENCES spacecraft_registry(spacecraft_id),
            name VARCHAR(100),
            contact_id VARCHAR(500),
            model_type VARCHAR(20) NOT NULL,
            
            mean_anomaly FLOAT,
            eccentricity FLOAT,
            inclination FLOAT,
            raan FLOAT,
            arg_perigee FLOAT,
            mean_motion FLOAT,
            doppler_bias_hz FLOAT,
            
            ssr FLOAT,
            aic FLOAT,
            bic FLOAT,
            chi_square_reduced FLOAT,
            metadata_json JSONB,             
            
            PRIMARY KEY (timestamp, spacecraft_id, model_type)
        );
        """)
        cursor.execute("SELECT create_hypertable('bls_orbit_determination', 'timestamp', if_not_exists => TRUE);")

        # ---------------------------------------------------------
        # TABLE 5: Processing Logs (for Grafana)
        # ---------------------------------------------------------
        print("Creating processing_logs table...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS processing_logs (
            timestamp TIMESTAMPTZ NOT NULL,
            tag VARCHAR(20) NOT NULL,
            code VARCHAR(10),
            message TEXT NOT NULL,
            source VARCHAR(100),
            contact_id VARCHAR(500),
            spacecraft_id VARCHAR(50),
            metadata_json JSONB
        );
        """)
        cursor.execute("SELECT create_hypertable('processing_logs', 'timestamp', if_not_exists => TRUE);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_tag ON processing_logs (tag, timestamp DESC);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_source ON processing_logs (source, timestamp DESC);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_contact ON processing_logs (contact_id, timestamp DESC);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_spacecraft ON processing_logs (spacecraft_id, timestamp DESC);")

        print("Database initialization complete.")

    except Exception as exc:
        print(f"An error occurred: {exc}")
        raise
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize the DASH TimescaleDB schema")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="drop existing DASH tables before recreating them (destructive)",
    )
    args = parser.parse_args()
    init_schema(reset=args.reset)
