from dataclasses import dataclass
from lib.utils import (
    _safe_str,
    _safe_bool,
    _join_list,
)

@dataclass
class TrackingContext:
    """
    Encapsulates and normalizes the incoming Grafana payload.
    """
    # Query Mode 1
    spacecraft_uuid: str | None
    start_time_iso: str | None 
    end_time_iso: str | None
    
    # Query Mode 2
    contact_uuid_list: str | None
    
    # Query Mode 3
    ephemeris_id: str | None
    
    # Other
    mode: str 
    lock_requirement: bool

    # Database filter parameters
    min_elevation: float = 1.0
    minimum_ebn0: float = 0.0
    min_doppler: float = 1.0
    max_doppler: float = 1e5
    
    # Optimization modes (Mapped 1:1 with internal Config)
    angle_constraint: float = 5.0      
    penalty_weight: float = 1e3        
    qmc_samples: int = 64              
    loss_function: str = "cauchy"      
    f_scale: float = 500.0             
    model_type: str = "auto"           
    criterion: str = "bic"             
    use_qmc: bool = True               
    method: str = "dogbox"             

    @classmethod
    def from_payload(cls, payload: dict):
        kwargs = {
            "spacecraft_uuid": _safe_str(payload.get("spacecraft_UUID")),
            "start_time_iso": _safe_str(payload.get("start_time")),
            "end_time_iso": _safe_str(payload.get("end_time")),
            "contact_uuid_list": _join_list(payload.get("contact_UUID_List")),
            "ephemeris_id": _safe_str(payload.get("EphemerisID")),
            "lock_requirement": _safe_bool(payload.get("lockRequirement")),
            "mode": str(payload.get("Mode", ""))
        }

        optional_fields = {
            "min_elevation": ("minElevation", float),
            "minimum_ebn0": ("minimumEbN0", float),
            "min_doppler": ("minDoppler", float),
            "max_doppler": ("maxDoppler", float),
            "angle_constraint": ("angleConstraint", float),
            "penalty_weight": ("penaltyWeight", float), 
            "qmc_samples": ("qmcSamples", int),
            "loss_function": ("lossFunction", str),
            "f_scale": ("fScale", float),
            "model_type": ("modelType", str),
            "criterion": ("criterion", str),
            "use_qmc": ("useQmc", _safe_bool),
            "method": ("method", str)
        }

        for dc_field, (payload_key, cast_func) in optional_fields.items():
            val = payload.get(payload_key)
            if val is not None:
                casted_val = cast_func(val)
                if casted_val == "":
                    continue
                kwargs[dc_field] = casted_val

        return cls(**kwargs)
