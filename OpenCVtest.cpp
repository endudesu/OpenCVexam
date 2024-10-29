#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <functional>

using namespace cv;
using namespace std;



void HelloCV();
void MatOp2();
void MatOp3();
void MatOp4();
void Project1();
void brightness();
void HistImage();
//void Q1();


void runMenu(const vector<pair<string, function<void()>>>& functions);

int main() {
    vector<pair<string, function<void()>>> functions = {
        {"HelloCV - Display OpenCV version and image", HelloCV},
        {"MatOp2 - Perform operations on dog image", MatOp2},
        {"MatOp3 - Perform operations on cat image", MatOp3},
        {"MatOp4 - Increment pixel values in cat image", MatOp4},
        {"Project1", Project1},
        {"brightness", brightness},
        {"HistImage", HistImage}

        // New function added here
    };

    runMenu(functions);

    return 0;
}

void runMenu(const vector<pair<string, function<void()>>>& functions) {
    int choice;

    do {
        cout << "\nSelect an option:" << endl;
        for (size_t i = 0; i < functions.size(); ++i) {
            cout << i + 1 << ". " << functions[i].first << endl;
        }
        cout << functions.size() + 1 << ". Exit" << endl;

        cin >> choice;

        if (cin.fail()) {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "Invalid input. Please enter a number." << endl;
        }
        else if (choice > 0 && choice <= functions.size()) {
            functions[choice - 1].second();
        }
        else if (choice != functions.size() + 1) {
            cout << "Invalid choice. Please try again." << endl;
        }
    } while (choice != functions.size() + 1);

    cout << "Exiting..." << endl;
}

void HelloCV() {
    cout << "Hello OpenCV " << CV_VERSION << endl;

    cv::Mat img = cv::imread("lenna.bmp");
    if (img.empty()) {
        cerr << "Image load failed" << endl;
        return;
    }

    namedWindow("image");
    imshow("image", img);
    waitKey(0);
    destroyAllWindows();
}

void MatOp2() {
    Mat img1 = imread("dog.bmp");
    if (img1.empty()) {
        cerr << "Image load failed!" << endl;
        return;
    }

    Mat img2 = img1;
    Mat img3 = img1;
    Mat img4 = img1.clone();
    Mat img5;
    img1.copyTo(img5);

    img1.setTo(Scalar(0, 255, 255));

    imshow("img1", img1);
    imshow("img2", img2);
    imshow("img3", img3);
    imshow("img4", img4);
    imshow("img5", img5);
    waitKey();
    destroyAllWindows();
}

void MatOp3() {
    Mat img1 = imread("cat.bmp");
    if (img1.empty()) {
        cerr << "Image load failed!" << endl;
        return;
    }

    Mat img2 = img1(Rect(220, 120, 340, 240));
    Mat img3 = img1(Rect(220, 120, 340, 240)).clone();
    img2 = ~img2;

    imshow("img1", img1);
    imshow("img2", img2);
    imshow("img3", img3);
    waitKey();
    destroyAllWindows();
}

void MatOp4() {
    Mat img1 = imread("cat.bmp");

    if (img1.empty()) {
        cerr << "Image load failed!" << endl;
        return;
    }

    Mat img2 = img1.clone();
    Mat img3 = img1.clone();
    Mat img4 = img1.clone();

    for (int y = 0; y < img2.rows; y++) {
        for (int x = 0; x < img2.cols; x++) {
            Vec3b& pixel = img2.at<Vec3b>(y, x);
            for (int c = 0; c < 3; c++) {
                pixel[c] = saturate_cast<uchar>(pixel[c] + 50);  
            }
        }
    }

    for (int y = 0; y < img3.rows; y++) {
        Vec3b* rowPtr = img3.ptr<Vec3b>(y);
        for (int x = 0; x < img3.cols; x++) {
            for (int c = 0; c < 3; c++) {
                rowPtr[x][c] = saturate_cast<uchar>(rowPtr[x][c] + 50);  
            }
        }
    }

    for (MatIterator_<Vec3b> it = img4.begin<Vec3b>(); it != img4.end<Vec3b>(); ++it) {
        for (int c = 0; c < 3; c++) {
            (*it)[c] = saturate_cast<uchar>((*it)[c] + 50);  
        }
    }

    imshow("Img1", img1);
    imshow("Img2", img2);
    imshow("Img3", img3);
    imshow("Img4", img4);

    waitKey(0);
    destroyAllWindows();

}

void Project1() {
    Mat img1 = imread("cat.bmp");

    if (img1.empty()) {
        cerr << "Image load failed!" << endl;
        return;
    }

    Mat img2 = img1.clone();

    for (int y = 0; y < img2.rows; y++) {
        for (int x = 0; x < img2.cols; x++) {
            Vec3b& pixel = img2.at<Vec3b>(y, x);
            uchar average = (pixel[0] + pixel[1] + pixel[2]) / 3; 
            average = (average > 255) ? 255 : (average < 0) ? 0 : average;
            pixel = Vec3b(average, average, average);
        }
    }
    imshow("img1", img1);
    imshow("img2", img2);

    waitKey();
    destroyAllWindows();
}

void brightness() {
    Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

    if (src.empty()) {
        cerr << "Image load failed!" << endl;
        return;
    }

    Mat dst(src.rows, src.cols, src.type());
    
    for (int j = 0; j < src.rows; j++) {
        for (int i = 0; i < src.cols; i++) {
            dst.at<uchar>(j, i) = src.at<uchar>(j, i) + 100;
        }
    }

    imshow("srv", src);
    imshow("dst", dst);
    waitKey();

    destroyAllWindows();
}

Mat calcGrayHist(const Mat& img) {

    CV_Assert(img.type() == CV_8UC1);

    Mat hist;
    int channels[] = { 0 };
    int dims = 1;
    const int histSize[] = { 256 };
    float graylevel[] = { 0,256 };
    const float* ranges[] = { graylevel };

    calcHist(&img, 1, channels, noArray(), hist, dims, histSize, ranges);

    return hist;
}

Mat getGrayHistImage(const Mat& hist) {

    CV_Assert(hist.type() == CV_32FC1);
    CV_Assert(hist.size() == Size(1, 256));

    double histMax;
    minMaxLoc(hist, 0, &histMax);

    Mat imgHist(100, 256, CV_8UC1, Scalar(255));
    for (int i = 0; i < 256; i++) {
        line(imgHist, Point(i, 100),
            Point(i, 100 - cvRound(hist.at<float>(i, 0) * 100 / histMax)), Scalar(0));
    }

    return imgHist;
}

void HistImage() {
    Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return;
    }
    Mat hist = calcGrayHist(src); 
    Mat hist_img = getGrayHistImage(hist);

    imshow("src", src); 
    imshow("srcHist", hist_img);
    waitKey(0); 
}

/*
int main() {
    string choice1;
    cout << "choice RGB" << endl;
    cin >> choice1;

    Mat img = imread("cat.bmp");
    vector<Mat> channels;
    split(img, channels);
    while(1) {
        if (choice1 == "r") {
            imshow("r", channels[2]);
            waitKey();
            choice1.clear();
        }
        else if (choice1 == "g") {
            imshow("g", channels[1]);
            waitKey();
            destroyAllWindows();

        }
        else if (choice1 == "b") {
            imshow("b", channels[0]);
            waitKey();
            destroyAllWindows();

        }
        else if (choice1 == "c") {
            imshow("c", channels);
            waitKey();
            destroyAllWindows();

        }
        else(cout << "err" << endl);
    }

}
/*
string title = "main";
Mat image;
Point p1(0, 0);
Point p2(0, 0);

void showNewImg();


void onMouse(int event, int x, int y, int flags, void* param) {
    switch (event) {
    case EVENT_LBUTTONDOWN:
        p1.x = x;
        p1.y = y;
        p2.x = x;
        p2.y = y;
        break;
    case EVENT_LBUTTONUP:
        p2.x = x;
        p2.y = y;
        showNewImg();

    default:
        break;
    }
}


int main() {
    image = imread("Lenna.bmp", IMREAD_COLOR);
    CV_Assert(image.data);

    imshow(title, image);

    setMouseCallback(title, onMouse, 0);
    waitKey();

    return 0;

}

void showNewImg() {//관심영역이 맞는 새로운 윈도우를 표시
    int wdth = p2.x - p1.x;
    int hgt = p2.y - p1.y;
    Rect roi(p1.x, p1.y, wdth, hgt);//관심영역 
    
    Mat draggedImage = image.clone();//호출 될 때 마다 드래그된 이미지의 깊은 복사
    //클릭시마다 호출이니 클릭마다 반전영역을 중첩시키지 않는다는 뜻입니다.
    for (MatIterator_<Vec3b> it = draggedImage.begin<Vec3b>(); it != draggedImage.end<Vec3b>(); ++it) {
        for (int c = 0; c < 3; c++) {
            (*it)[c] = saturate_cast<uchar>((*it)[c] + 50);
        }
    }
    Mat newimg = draggedImage(roi);//
    imshow(title, draggedImage);
}
*/