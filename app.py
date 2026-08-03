# ============================================================  
# CAPSTONE MAIN APPLICATION  
# Intelligent Healthcare Diagnostic Assistant  
# Introduction to AI — 13-Week Capstone  
# ============================================================  

import sys  
import json  
import warnings  
import numpy as np  
import matplotlib.pyplot as plt  
warnings.filterwarnings('ignore')  

# Import all modules  
from modules.agent          import HealthcareDiagnosticAgent, PatientPercept  
from modules.knowledge_base import MedicalKnowledgeBase  
from modules.bayesian_net   import SimpleBayesianDiagnostics  
from modules.ml_classifier  import MLDiagnosticClassifier  
from modules.neural_network import NeuralDiagnosticModel  
from modules.fuzzy_controller import FuzzySeverityAssessor  
from modules.planner        import TreatmentPlanner  

# ── ANSI Colors ────────────────────────────────────────────  
class C:  
    HEADER = '\033[95m'; BLUE   = '\033[94m'  
    GREEN  = '\033[92m'; YELLOW = '\033[93m'  
    RED    = '\033[91m'; BOLD   = '\033[1m'  
    END    = '\033[0m'  

def banner():  
    print(f"""  
{C.BOLD}{C.BLUE}  
╔══════════════════════════════════════════════════════════╗  
║        🏥 INTELLIGENT HEALTHCARE DIAGNOSTIC AI           ║  
║         Introduction to AI — Capstone Project            ║  
║  Modules: Agents | Logic | Bayes | ML | DNN | Fuzzy      ║  
╚══════════════════════════════════════════════════════════╝  
{C.END}""")  

def section(title: str):  
    print(f"\n{C.BOLD}{C.YELLOW}{'═'*60}{C.END}")  
    print(f"{C.BOLD}{C.YELLOW}  {title}{C.END}")  
    print(f"{C.BOLD}{C.YELLOW}{'═'*60}{C.END}")  

def build_system() -> HealthcareDiagnosticAgent:  
    """Instantiate and wire all AI modules"""  
    section("🔧 Building AI System — Registering Modules")  

    agent = HealthcareDiagnosticAgent()  

    print("\n  Initializing modules (Please wait while ML and DNN models train)...")  
    
    # Instantiate modules
    kb_module = MedicalKnowledgeBase()  
    bayes_module = SimpleBayesianDiagnostics()  
    ml_module = MLDiagnosticClassifier()  
    dnn_module = NeuralDiagnosticModel()  
    fuzzy_module = FuzzySeverityAssessor()  
    planner_module = TreatmentPlanner()  

    # Pre-train the machine learning models so they are ready for inference
    ml_module.train(verbose=False)
    dnn_module.train(epochs=30, verbose=0)

    # Register modules with the Intelligent Agent
    agent.register_module('Knowledge Base', kb_module)  
    agent.register_module('Bayesian Network', bayes_module)  
    agent.register_module('ML Classifier', ml_module)  
    agent.register_module('Deep Neural Net', dnn_module)  
    agent.register_module('Fuzzy Severity', fuzzy_module)  
    agent.register_module('Treatment Planner', planner_module)  

    return agent

def get_test_patients() -> list[PatientPercept]:
    """Generate 5 distinct test patients to satisfy project requirements"""
    return [
        PatientPercept(
            patient_id="P001",
            symptoms=["fever", "cough", "loss_of_smell", "fatigue"],
            age=34, temperature=38.9, heart_rate=98, blood_pressure="120/80"
        ),
        PatientPercept(
            patient_id="P002",
            symptoms=["chest_pain", "shortness_of_breath", "sweating"],
            age=65, temperature=37.2, heart_rate=115, blood_pressure="150/95"
        ),
        PatientPercept(
            patient_id="P003",
            symptoms=["frequent_urination", "excessive_thirst", "fatigue"],
            age=42, temperature=37.0, heart_rate=80, blood_pressure="130/85"
        ),
        PatientPercept(
            patient_id="P004",
            symptoms=["fever", "rash", "joint_pain", "headache"],
            age=28, temperature=39.5, heart_rate=105, blood_pressure="110/70"
        ),
        PatientPercept(
            patient_id="P005",
            symptoms=["headache", "stiff_neck", "light_sensitivity", "fever"],
            age=19, temperature=40.2, heart_rate=125, blood_pressure="100/65"
        )
    ]

def main():
    banner()
    
    agent = build_system()
    
    # Load the test patients
    section("🩺 Processing Test Patients")
    patients = get_test_patients()
    
    # Run the Perceive -> Think -> Act loop for each patient
    for idx, patient in enumerate(patients, 1):
        print(f"\n{C.BOLD}{C.BLUE}=== Patient {idx}/5: ID {patient.patient_id} ==={C.END}")
        print(f"  Vitals: Temp {patient.temperature}°C | HR {patient.heart_rate} bpm")
        print(f"  Symptoms: {', '.join(patient.symptoms)}")
        
        # Run the agent
        report = agent.run(patient)
        
        # Display aggregated results
        print(f"\n  {C.GREEN}Final Diagnosis:{C.END} {str(report['diagnosis']).upper()}")
        print(f"  {C.GREEN}Confidence:{C.END}    {report['confidence']:.2%}")
        print(f"  {C.GREEN}Urgency:{C.END}       {report['urgency']}")
        print(f"  {C.GREEN}Next Action:{C.END}   {report['next_action']}")
        
        print(f"\n  {C.YELLOW}Recommendations:{C.END}")
        for rec in report['recommendations']:
            print(f"    - {rec}")
            
        # NEW: Print the dynamic Treatment Plan
        print(f"\n  {C.YELLOW}Step-by-Step Treatment Plan:{C.END}")
        if 'treatment_plan' in report and 'plan' in report['treatment_plan']:
            if not report['treatment_plan']['plan']:
                print(f"    {C.RED}No valid medical plan found for {report['diagnosis']}{C.END}")
            else:
                for step in report['treatment_plan']['plan']:
                    print(f"    Step {step['step']:2d}: {step['action']:<30} [{step['duration']}]")
            
        print("─" * 60)

    # Print the internal agent logs to demonstrate state changes
    agent.print_log()
    
    # Print final performance evaluation
    perf = agent.get_performance()
    print(f"\n{C.BOLD}Total Patients Processed: {perf['total_patients']}{C.END}")
    print(f"{C.BOLD}Agent Performance Score: {perf['performance_score']}{C.END}")

if __name__ == "__main__":
    main()