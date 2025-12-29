"""
Update script paths to use FINAL_ prefixed directories
"""

import os
import re
from pathlib import Path

def update_script_paths():
    """Update paths in key evaluation and visualization scripts"""
    
    print("🔧 Updating script paths to use FINAL_ directories...")
    
    # Scripts to update
    scripts_to_update = [
        'evaluate_stage2_gt.py',
        'evaluate_20_test_images.py',
        'visualize_risk_zones_vit.py',
        'visualize_improved_risk_zones_v2.py',
        'visualize_repo_stats.py',
        'generate_repo_stats.py'
    ]
    
    # Path replacements
    replacements = {
        'outputs/risk_zones_vit': 'outputs/FINAL_risk_zones_vit',
        'outputs/improved_risk_zones_v2': 'outputs/FINAL_improved_risk_zones_v2',
        'outputs/visual_evaluation': 'outputs/FINAL_visual_evaluation',
        'outputs/repo_visualizations': 'outputs/FINAL_repo_visualizations',
        'runs/vit_classifier': 'runs/FINAL_vit_classifier',
        'runs/pipeline_optimization': 'runs/FINAL_pipeline_optimization',
        'runs/full_pipeline_validation': 'runs/FINAL_full_pipeline_validation',
        "'risk_zones_vit'": "'FINAL_risk_zones_vit'",
        '"risk_zones_vit"': '"FINAL_risk_zones_vit"',
        "'improved_risk_zones_v2'": "'FINAL_improved_risk_zones_v2'",
        '"improved_risk_zones_v2"': '"FINAL_improved_risk_zones_v2"',
    }
    
    updates_made = 0
    
    for script_name in scripts_to_update:
        script_path = Path(script_name)
        
        if not script_path.exists():
            print(f"   ⚠️  Script not found: {script_name}")
            continue
        
        # Read script content
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply replacements
        for old_path, new_path in replacements.items():
            if old_path in content:
                content = content.replace(old_path, new_path)
                updates_made += 1
        
        # Write back if changed
        if content != original_content:
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"   ✅ Updated: {script_name}")
    
    print(f"\n📊 Total path updates: {updates_made}")
    print("✅ Scripts now use FINAL_ directories!")

if __name__ == '__main__':
    update_script_paths()
