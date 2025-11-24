import torch
from models.network_swinir import SwinIR as TeacherModel
from models.network_swinir_student import SwinIR_Student as StudentModel

def count_parameters(model):
    """A helper function to count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # --- Define the Teacher Model Architecture ---
    # These parameters match the 'classical_sr' medium model
    teacher_args = {
        'upscale': 4,
        'img_size': 64,
        'window_size': 8,
        'embed_dim': 180,
        'depths': [6, 6, 6, 6, 6, 6],
        'num_heads': [6, 6, 6, 6, 6, 6]
    }
    teacher = TeacherModel(**teacher_args)
    teacher_params = count_parameters(teacher)

    # --- Define our new Student Model Architecture ---
    # These are the parameters we are proposing for our lightweight model
    student_args = {
        'upscale': 4,
        'img_size': 64,
        'window_size': 8,
        'embed_dim': 60,
        'depths': [4, 4, 4, 4],
        'num_heads': [6, 6, 6, 6]
    }
    student = StudentModel(**student_args)
    student_params = count_parameters(student)

    # --- Print the results ---
    print("--- Model Size Comparison ---")
    print(f"Teacher Model Parameters: {teacher_params:,}")
    print(f"Student Model Parameters: {student_params:,}")
    
    percentage = (student_params / teacher_params) * 100
    print(f"\nStudent model is {percentage:.2f}% the size of the Teacher model.")

if __name__ == '__main__':
    main()