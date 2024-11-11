# imports
import os
import cv2  # pip install opencv-python
import time



# function to load images from a specified folder
def load_images_from_folder(folder, skip_dirs=None):
    # initialize empty list to store the images and their paths
    images = []
    
    # if no directories to skip are specified, initialize to an empty list
    if skip_dirs is None:
        skip_dirs = []
    
    # walking/going through the folder, including all subfolders
    for root, _, files in os.walk(folder):
        # skip the root folder - this makes sure to save only the images in the subfolders/classes
        # if root == folder:
        #     continue  # Skip files directly in the root folder
        
        # skip any directories specified in skip_dirs
        if any(skip in root for skip in skip_dirs):
            continue  # Skip this directory and its contents
        
        # iterate over each file in the folder
        for file in files:
            # check if the file has an image extension
            if file.endswith(('jpg', 'jpeg', 'png')):
                img_path = os.path.join(root, file)         # Form the full file path
                img = cv2.imread(img_path)                  # Read image using OpenCV
                
                if img is not None:                         # Ensure image was loaded successfully
                    images.append((img, img_path))          # Append image with its path as a tuple
                
    # return images list
    return images

def list_classes(dataset):
    # define a list to store the classes
    classes = []
    
    # iterate over each file in the folder
    for dir in os.listdir(dataset):
        # if dir does not end with an image extension
        if not dir.endswith(('jpg', 'jpeg', 'png')):
            # display class names
            print(dir)
            
            # append the dir to the classes list
            classes.append(dir)
            
    # return list
    return classes

def menu(dataset):
    # display menu
    print("----- <Menu> -----")
    valid_classes = list_classes(dataset)
    
    # loop until valid input is received
    while True:
        response = input("\nEnter the name of the constellation you want to search for: ").strip().capitalize()
        
        # check if input is a valid class name
        if response in valid_classes:
            # return valid user input
            return response
        else:
            # display error message
            print("!!! <Invalid class name. Please try again.> !!!")

def template_matching(template_imgs, target_img):
    # parameters
    results = []
    output_path = "Results_TemplateMatching"                # path to store the output images
    # check if the output folder exists else create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for template_img, template_path in template_imgs:
        # convert the template image to grayscale
        template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        # get the height and width of the template image
        height, width = template_img.shape

        # list of methods to be used for template matching
        # suggested by OpenCV documentation since they have different accuracies
        methods_list = [(cv2.TM_CCOEFF, 'TM_CCOEFF'), (cv2.TM_CCOEFF_NORMED, 'TM_CCOEFF_NORMED'), 
                        (cv2.TM_CCORR, 'TM_CCORR'), (cv2.TM_CCORR_NORMED, 'TM_CCORR_NORMED'), 
                        (cv2.TM_SQDIFF, 'TM_SQDIFF'), (cv2.TM_SQDIFF_NORMED, 'TM_SQDIFF_NORMED')]

        # iterate over each method
        for method, method_name in methods_list:
            # create a copy of the target image
            target_image_copy = target_img.copy()
            
            # start timer
            start_time = time.time()
            
            # perform template matching by convolution
            result = cv2.matchTemplate(target_image_copy, template_img, method)
            
            # end timer
            end_time = time.time()
            
            # calculate time taken
            time_taken = end_time - start_time
            
            # obtain the min and max values and their locations
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # if the method is TM_SQDIFF or TM_SQDIFF_NORMED, use min_loc
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc  # top left corner
                score = min_val     # score 
            else:
                top_left = max_loc  # top left corner
                score = max_val     # score
            # calculate the bottom right corner
            bottom_right = (top_left[0] + width, top_left[1] + height)
            
            # store the result
            results.append({
                'template': os.path.basename(template_path), 
                'method': method_name, 
                'score': score,
                # round the time taken to 4 decimal places
                'time_taken': round(time_taken, 4),  
            })
            
            # draw a rectangle around the matched template
            cv2.rectangle(target_image_copy, top_left, bottom_right, 255, 2)
            
            # construct the filename based on the template name and method name
            template_name = os.path.splitext(os.path.basename(template_path))[0]
            output_filename = f"{template_name}_{method_name}.png"
            output_file_path = os.path.join(output_path, output_filename)

            # save the output image
            cv2.imwrite(output_file_path, target_image_copy)
            print(f"----- <Processed {output_filename} - Score: {round(score, 2)} - TimeTaken: {time_taken:.4f}s> -----")
            
    print("\n----- <Done> -----\n")
    
    # sort the results based on the score in descending order
    top_results = sorted(results, key=lambda x: x['score'], reverse=True)[:5]
    
    # display the top 5 matches
    print("----- <Top Matches> -----")
    for res in top_results:
        print(f"Template: {res['template']} - Method: {res['method']} - Score: {round(res['score'], 2)} - TimeTaken: {res['time_taken']}s\n")



def load_target_image(target_image_path, class_name):
    # load the target image as a grayscale image
    target_image = cv2.imread(target_image_path, 0)
    
    # check if target_image is not None
    if target_image is not None:
        if class_name in ['Gemini', 'Orion', 'Canis Major', 'Taurus', 'Cassiopeia', 'Cygnus'] and not os.path.basename(target_image_path).split('.')[0] in ['targetImage1', 'targetImage1-NoLine']:
            # displaying an error message
            print("!!! <Target Image Invalid. Please Use 'targetImage2' or 'targetImage2-NoLine'> !!!")
            return None
            
        elif class_name in ['Scorpius', 'Libra', 'Leo', 'Cancer']:
            # displaying an error message
            print("!!! <Target Image Invalid. Please Use 'targetImage1' or 'targetImage1-NoLine'> !!!")
            return None
            
        # displaying a completion message
        print("----- <Target Image Found and Loaded Successfully> -----\n")
        
        # display target image
        print("----- <Displaying Target Image> -----")
        cv2.imshow("Target Image", target_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("----- <Target Image Displayed Successfully> -----\n")
        
    # return the target image
    return target_image


def main():
    # parameters
    dataset_path = "ConstellationDataset"                                  # path to the dataset
    target_image_path = f"{dataset_path}/TargetImages/targetImage1.png"      # path to the target image
    skip_dirs = ["TargetImages"]                                             # directories to skip

    # check if the dataset exists
    if os.path.exists(dataset_path):
        # print completion message
        print("----- <Dataset Found and Loaded Successfully> -----")
        
        # get target image
        target_image = load_target_image(target_image_path)
        
        # check if target image is not None
        if target_image is not None:
            # call menu function
            # this is done such that only the chosen constellation will be seached for effectivly reducing computation time and resources
            userInput = menu(dataset_path)
            
            # contruct the path to the chosen constellation
            class_path = os.path.join(dataset_path, userInput)
            
            # obtain the templates for the chosen constellation
            template_imgs = load_images_from_folder(class_path, skip_dirs)
            print(f"----- <Loaded {len(template_imgs)} images from the dataset> -----\n")
            
            # call the template matching function
            template_matching(template_imgs, target_image)
            
        else:
            # displaying an error message
            print("!!! <Target Image Not Found> !!!")
    else:
        # displaying an error message
        print("!!! <Dataset Not Found> !!!")
        
    # cv2 cleanup
    cv2.destroyAllWindows()
    



if __name__ == "__main__":
    main()

'''
Refereces:

- https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html - OpenCV Documentation
- https://www.youtube.com/watch?v=T-0lZWYWE9Y&t=276s - OpenCV Python Tutorial #7 - Template Matching (Object Detection)
- https://www.youtube.com/watch?v=cDP_4VbC_sE&t=233s - Tips Tricks 25 - Locating objects in large images using template matching
- https://www.youtube.com/watch?v=PqVH9IHMLss - Tutorial 58 - Object detection using template matching
- https://www.youtube.com/watch?v=LXS2ter_5ds - OpenCV 21: Template Matching | Python | OpenCV

Please Note: The YouTube videos were mostly used to understand what is happening underneath the OpenCV functions
'''