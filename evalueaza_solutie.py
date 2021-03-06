def evaluate_results_task1(predictions_path,ground_truth_path,verbose = 0):
    total_correct = 0
    for i in range(1,21):
        add_0 = ''
        if i <= 9:
        	add_0 = '0'
        filename_predictions = predictions_path + "/" + add_0 + str(i) + "_predicted.txt"
        p = open(filename_predictions,"rt")
        filename_ground_truth = ground_truth_path + "/" + add_0 + str(i) + "_gt.txt"
        gt = open(filename_ground_truth,"rt")
        correct_flag = 1
        for row in range(1,10):
            p_line = p.readline()
            gt_line = gt.readline()
            #print(p_line)
            #print(gt_line)
            if (p_line[:10] != gt_line[:10]):
                correct_flag = 0
        p.close()
        gt.close()
        
        if verbose:
            print("Task 1 - Classic Sudoku: for test example number ", str(i), " the prediction is :", (1-correct_flag) * "in" + "correct", "\n")
        
        total_correct = total_correct + correct_flag
        points = total_correct * 0.2


    # total_correct_bonus = 0
    # for i in range(1,21):
    #     filename_predictions = predictions_path + "/" + str(i) + "_bonus_predicted.txt"
    #     p = open(filename_predictions,"rt")
    #     add_0 = ''
    #     if i <= 9:
    #     	add_0 = '0'
    #     filename_ground_truth = ground_truth_path + "/" + add_0 + str(i) + "_bonus_gt.txt"
    #     gt = open(filename_ground_truth,"rt")
    #     correct_flag = 1
    #     for row in range(1,10):
    #         p_line = p.readline()
    #         gt_line = gt.readline()
    #         #print(p_line)
    #         #print(gt_line)
    #         if (p_line[:10] != gt_line[:10]):
    #             correct_flag = 0
    #     p.close()
    #     gt.close()
    #
    #     if verbose:
    #         print("Task 1 bonus - Classic Sudoku: for test example number ", str(i), " the prediction is :", (1-correct_flag) * "in" + "correct", "\n")
    #
    #     total_correct_bonus = total_correct_bonus + correct_flag
    #     points_bonus = total_correct_bonus * 0.025

    return total_correct, points

def evaluate_results_task2(predictions_path,ground_truth_path,verbose = 0):
    total_correct = 0
    for i in range(1,41):
        add_0 = ''
        if i <= 9:
            add_0 = '0'
        filename_predictions = predictions_path + "/" + add_0 + str(i) + "_predicted.txt"
        p = open(filename_predictions,"rt")        

        filename_ground_truth = ground_truth_path + "/" + add_0 + str(i) + "_gt.txt"
        gt = open(filename_ground_truth,"rt")
        correct_flag = 1
        for row in range(1,10):
            p_line = p.readline()
            gt_line = gt.readline()
            #print(p_line)
            #print(gt_line)
            if (p_line[:19] != gt_line[:19]):
                correct_flag = 0
                print(i, row)
        p.close()
        gt.close()
        
        if verbose:
            print("Task 2 - Jigsaw Sudoku: for test example number ", str(i), " the prediction is :", (1-correct_flag) * "in" + "correct", "\n")
        
        total_correct = total_correct + correct_flag
        points = total_correct * 0.1
    
	
    # total_correct_bonus = 0
    # for i in range(1,41):
    #     filename_predictions = predictions_path + "/" + str(i) + "_bonus_predicted.txt"
    #     p = open(filename_predictions,"rt")
    #     add_0 = ''
    #     if i <= 9:
    #     	add_0 = '0'
    #     filename_ground_truth = ground_truth_path + "/" + add_0 + str(i) + "_bonus_gt.txt"
    #     gt = open(filename_ground_truth,"rt")
    #     correct_flag = 1
    #     for row in range(1,10):
    #         p_line = p.readline()
    #         gt_line = gt.readline()
    #         #print(p_line)
    #         #print(gt_line)
    #         if (p_line[:19] != gt_line[:19]):
    #             correct_flag = 0
    #     p.close()
    #     gt.close()
    #
    #     if verbose:
    #         print("Task 2 bonus - Sudoku Jigsaw: for test example number ", str(i), " the prediction is :", (1-correct_flag) * "in" + "correct", "\n")
    #
    #     total_correct_bonus = total_correct_bonus + correct_flag
    #     points_bonus = total_correct_bonus * 0.025
                
    return total_correct, points


def evaluate_results_bonus(predictions_path,ground_truth_path, verbose = 0):
	pass


verbose = 0

#change this on your machine
predictions_path_root = "predicted/"
ground_truth_path_root = "antrenare/"

#task1
predictions_path = predictions_path_root + "clasic"
ground_truth_path = ground_truth_path_root + "clasic"
total_correct_task1,points_task1 = evaluate_results_task1(predictions_path,ground_truth_path,verbose)
print(total_correct_task1,points_task1)

#task2
predictions_path = predictions_path_root + "jigsaw"
ground_truth_path = ground_truth_path_root + "jigsaw"
total_correct_task2,points_task2 = evaluate_results_task2(predictions_path,ground_truth_path,verbose)
print(total_correct_task2,points_task2)