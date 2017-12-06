#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <vector>    // std::vector

using namespace cv;

int main(int argc, char** argv)
{
    // READ RGB color image and convert it to Lab
    cv::Mat bgr_img = cv::imread("sauvc2.png");
    cv::Mat lab_image,bgr_image;
   GaussianBlur(bgr_img,bgr_image,Size(1,1),2,2); 
    cv::cvtColor(bgr_image, lab_image, CV_BGR2Lab);

    // Extract the L channel
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

    // apply the CLAHE algorithm to the L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    cv::Mat dst;
    clahe->apply(lab_planes[0], dst);

    // Merge the the color planes back into an Lab image
    dst.copyTo(lab_planes[0]);
    cv::merge(lab_planes, lab_image);

   // convert back to RGB
   cv::Mat image_clahe;
   cv::cvtColor(lab_image, image_clahe, CV_Lab2BGR);

   // Normalise the CLAHE-RGB result
    IplImage* image_clahe1=&image_clahe;
   //IplImage* image_clahe1 = cvCloneImage( &(IplImage)image_clahe ); 

    IplImage* redchannel = cvCreateImage(cvGetSize(image_clahe1), 8, 1);
    IplImage* greenchannel = cvCreateImage(cvGetSize(image_clahe1), 8, 1);
    IplImage* bluechannel = cvCreateImage(cvGetSize(image_clahe1), 8, 1);

    IplImage* redavg = cvCreateImage(cvGetSize(image_clahe1), 8, 1);
    IplImage* greenavg = cvCreateImage(cvGetSize(image_clahe1), 8, 1);
    IplImage* blueavg= cvCreateImage(cvGetSize(image_clahe1), 8, 1);
    IplImage* imgavg = cvCreateImage(cvGetSize(image_clahe1), 8, 3);

    cvSplit(image_clahe1, bluechannel, greenchannel, redchannel, NULL);

for(int x=0;x<image_clahe1->width;x++)
    {
        for(int y=0;y<image_clahe1->height;y++)
        {
            int redValue = cvGetReal2D(redchannel, y, x);
            int greenValue = cvGetReal2D(greenchannel, y, x);
            int blueValue = cvGetReal2D(bluechannel, y, x);
            double sum = redValue+greenValue+blueValue;
            cvSetReal2D(redavg, y, x, redValue/sum*255);
            cvSetReal2D(greenavg, y, x, greenValue/sum*255);
            cvSetReal2D(blueavg, y, x, blueValue/sum*255);
        }
    }

    cvMerge(blueavg, greenavg, redavg, NULL, imgavg);

    cv::Mat imgavg1=cvarrToMat(imgavg);

    cvReleaseImage(&redchannel);
    cvReleaseImage(&greenchannel);
    cvReleaseImage(&bluechannel);
    cvReleaseImage(&redavg);
    cvReleaseImage(&greenavg);
    cvReleaseImage(&blueavg);
    cvReleaseImage(&imgavg);

    

// display the results  (you might also want to see lab_planes[0] before and after).
   cv::imshow("image original", bgr_image);
   cv::imshow("image CLAHE", image_clahe);
   cv::imshow("image CLAHE normalise", imgavg1);
   cv::waitKey();
}
/*

int main(int argc, char** argv)
{
    // READ RGB color image and convert it to Lab
    cv::Mat inputImage = cv::imread("sauvc2.png");
if(inputImage.channels() >= 3)
    {
        Mat ycrcb;

        cvtColor(inputImage,ycrcb,CV_BGR2YCrCb);

        vector<Mat> channels;
        split(ycrcb,channels);

        equalizeHist(channels[0], channels[0]);

        Mat result;
        merge(channels,ycrcb);

        cvtColor(ycrcb,result,CV_YCrCb2BGR);
	
	 cv::imshow("image original",inputImage);
  	 cv::imshow("image HE", result);

        }

	
  	 cv::waitKey();
}
*/
    
