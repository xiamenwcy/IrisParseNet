/*
 * manuseg.cpp
 *
 *  Created on: 06.08.2012
 *      Author: Wild
 */

#include <map>
#include <list>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/regex.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace std;
using namespace cv;

/** no globbing in win32 mode **/
int _CRT_glob = 0;

/** Program modes **/
static const int MODE_ERROR = 0, MODE_MAIN = 1, MODE_HELP = 2;

/*
 * Print command line usage for this program
 */
void printUsage() {
	cout << "+-----------------------------------------------------------------------------+" << endl;
	cout << "| manuseg - manual iris segmentation                                          |" << endl;
	cout << "|                                                                             |" << endl;
	cout << "| MODES                                                                       |" << endl;
	cout << "|                                                                             |" << endl;
	cout << "| (# 1) applies manual segmentation / normalization                           |" << endl;
	cout << "| (# 2) usage                                                                 |" << endl;
	cout << "|                                                                             |" << endl;
    cout << "| ARGUMENTS                                                                   |" << endl;
    cout << "|                                                                             |" << endl;
    cout << "+------+------------+---+---+-------------------------------------------------+" << endl;
    cout << "| Name | Parameters | # | ? | Description                                     |" << endl;
    cout << "+------+------------+---+---+-------------------------------------------------+" << endl;
    cout << "| -i   | infile     | 1 | N | source image (* = any)                          |" << endl;
    cout << "| -c   | innerfile  | 1 | N | target coordinate files (?n = n-th * in infile) |" << endl;
    cout << "|      | outerfile  |   |   | with one point (x y) per line, e.g. 12 314      |" << endl;
    cout << "|      | upperfile  |   |   | for inner boundary, outer boundary, upper eyelid|" << endl;
    cout << "|      | lowerfile  |   |   | lower eyelid                                    |" << endl;
    cout << "| -o   | outfile    | 1 | N | output iris texture image                       |" << endl;
    cout << "| -m   | maskfile   | 1 | Y | write noise mask (off)                          |" << endl;
    cout << "| -sr  | segresfile | 1 | Y | write segmentation result (off)                 |" << endl;
    cout << "| -srl | segresfile | 1 | Y | write lidless segmentation result (off)         |" << endl;
    cout << "| -roi | roifile    | 1 | Y | write ROI result (off)                          |" << endl;
    cout << "| -s   | wdth hght  | 1 | Y | size, i.e. width and height, of output (512x64) |" << endl;
    cout << "| -e   |            | 1 | Y | enhance iris texture on (off)                   |" << endl;
    cout << "| -q   |            | 1 | Y | quiet mode on (off)                             |" << endl;
    cout << "| -t   |            | 1 | Y | time progress on (off)                          |" << endl;
    cout << "| -h   |            | 2 | N | prints usage                                    |" << endl;
    cout << "+------+------------+---+---+-------------------------------------------------+" << endl;
    cout << "|                                                                             |" << endl;
    cout << "| AUTHOR                                                                      |" << endl;
    cout << "|                                                                             |" << endl;
    cout << "| Peter Wild (pwild@cosy.sbg.ac.at)                                           |" << endl;
    cout << "|                                                                             |" << endl;
    cout << "| COPYRIGHT                                                                   |" << endl;
    cout << "|                                                                             |" << endl;
    cout << "| (C) 2012 All rights reserved. Do not distribute without written permission. |" << endl;
    cout << "+-----------------------------------------------------------------------------+" << endl;
}

/** ------------------------------- Clahe ------------------------------- **/

/**
 * Calculate a standard uniform upper exclusive lower inclusive 256-bin histogram for range [0,256]
 *
 * src: CV_8UC1 image
 * histogram: CV_32SC1 1 x 256 histogram matrix
 */
void hist2u(const Mat& src, Mat& histogram){
	histogram.setTo(0);
	MatConstIterator_<uchar> s = src.begin<uchar>();
	MatConstIterator_<uchar> e = src.end<uchar>();
	int * p = (int *)histogram.data;
	for (; s!=e; s++){
		p[*s]++;
	}
}

/*
 * Retrieves the (bilinear) interpolated byte from 4 bytes
 *
 * x: distance to left byte
 * y: distance to right byte
 * r: distance to upper byte
 * s: distance to lower byte
 * b1: upper left byte
 * b2: upper right byte
 * b3: lower left byte
 * b4: lower right byte
 */
uchar interp(const double x, const double y, const double r, const double s, const uchar b1, const uchar b2, const uchar b3, const uchar b4) {
  double w1 = (x + y);
  double w2 = x / w1;
  w1 = y / w1;
  double w3 = (r + s);
  double w4 = r / w3;
  w3 = s / w3;
  return saturate_cast<uchar>(w3 * (w1 * b1 + w2 * b2) + w4 * (w1 * b3 + w2 * b4));
}

/*
 * Retrieves the bilinear interpolated byte from 2 bytes
 *
 * x:  distance to left byte
 * y:  distance to right byte
 * b1: left byte
 * b2: right byte
 */
uchar interp(const double x, const double y, const uchar b1, const uchar b2) {
  double w1 = (x + y);
  double w2 = x / w1;
  w1 = y / w1;
  return saturate_cast<uchar>(w1 * b1 + w2 * b2);
}

/*
 * Inplace histogram clipping according to Zuiderveld (counts excess and redistributes excess by adding the average increment)
 *
 * hist: CV_32SC1 1 x 256 histogram matrix
 * clipFactor: between 0 (maximum slope M/N, where M #pixel in window, N #bins) and 1 (maximum slope M)
 * pixelCount: number of pixels in window
 */
void clipHistogram(Mat& hist, const float clipFactor, const int pixelCount) {
	double minSlope = ((double) pixelCount) / 256;
	int clipLimit = std::min(pixelCount, std::max(1, cvCeil(minSlope + clipFactor * (pixelCount - minSlope))));
	int distributeCount = 0;
	MatIterator_<int> p = hist.begin<int>();
	MatIterator_<int> e = hist.end<int>();
	for (; p!=e; p++){
		int binsExcess = *p - clipLimit;
		if (binsExcess > 0) {
			distributeCount += binsExcess;
			*p = clipLimit;
		}
	}
	int avgInc = distributeCount / 256;
	int maxBins = clipLimit - avgInc;
	for (p = hist.begin<int>(); p!=e; p++){
		if (*p <= maxBins) {
			distributeCount -= avgInc;
			*p += avgInc;
		}
		else if (*p < clipLimit) {
			distributeCount -= (clipLimit - *p);
			*p = clipLimit;
		}
	}
	while (distributeCount > 0) {
		for (p = hist.begin<int>(); p!=e && distributeCount > 0; p++){
			if (*p < clipLimit) {
				(*p)++;
				distributeCount--;
			}
		}
	}
}

/*
 * Contrast-limited adaptive histogram equalization (supports in-place)
 *
 * src: CV_8UC1 image
 * dst: CV_8UC1 image (in-place operation is possible)
 * cellWidth: patch size in x direction (greater or equal to 2)
 * cellHeight: patch size in y direction (greater or equal to 2)
 * clipFactor: histogram clip factor between 0 and 1
 */
void clahe(const Mat& src, Mat& dst, const int cellWidth = 10, const int cellHeight = 10, const float clipFactor = 1.){
	Mat hist(1,256,CV_32SC1);
	Mat roi;
	uchar * sp, * dp;
	int height = src.rows;
	int width = src.cols;
	int gridWidth = width / cellWidth + (width % cellWidth == 0 ? 0 : 1);
	int gridHeight = height / cellHeight + (height % cellHeight == 0 ? 0 : 1);
	int bufSize = (gridWidth + 2)*256;
	int bufOffsetLeft = bufSize - 256;
	int bufOffsetTop = bufSize - gridWidth * 256;
	int bufOffsetTopLeft = bufSize - (gridWidth + 1) * 256;
	Mat buf(1, bufSize, CV_8UC1);
	MatIterator_<uchar> pbuf = buf.begin<uchar>(), ebuf = buf.end<uchar>();
	MatIterator_<int> phist, ehist = hist.end<int>();
	uchar * curr, * topleft, * top, * left;
	int pixelCount, cX, cY, cWidth, cHeight, cellOrigin, cellOffset;
	double sum;
	// process first row, first cell
	cX = 0;
	cY = 0;
	cWidth = min(cellWidth, width);
	cHeight = min(cellHeight, height);
	pixelCount = cWidth*cHeight;
	sum = 0;
	roi = Mat(src,Rect(cX,cY,cWidth,cHeight));
	hist2u(roi,hist);
	if (clipFactor < 1) clipHistogram(hist,clipFactor,pixelCount);
	// equalization
	for(phist = hist.begin<int>(); phist!=ehist; phist++, pbuf++){
		sum += *phist;
		*pbuf = saturate_cast<uchar>(sum * 255 / pixelCount);
	}
	// paint first corner cell
	cWidth = min(cellWidth / 2, cWidth);
	cHeight = min(cellHeight / 2, cHeight);
	cellOrigin = src.step * cY + cX;
	cellOffset = src.step - cWidth;
	sp = (uchar *)(src.data + cellOrigin);
	dp = (uchar *)(dst.data + cellOrigin);
	curr = buf.data;
	for (int b=0; b < cHeight; b++, sp+= cellOffset, dp += cellOffset){
	  for (int a=0; a < cWidth; a++, sp++, dp++){
		*dp = curr[*sp];
	  }
	}
	// process first row, other cells
	for (int x = 1; x < gridWidth; x++) {
		cX = x*cellWidth;
		cWidth = min(cellWidth, width - x*cellWidth);
		cHeight = min(cellHeight, height);
		pixelCount = cWidth*cHeight;
		sum = 0;
		roi.release();
		roi = Mat(src,Rect(cX,cY,cWidth,cHeight));
		hist2u(roi,hist);
		if (clipFactor < 1) clipHistogram(hist,clipFactor,pixelCount);
		// equalization
		for(phist = hist.begin<int>(); phist!=ehist; phist++, pbuf++){
			sum += *phist;
			*pbuf = saturate_cast<uchar>(sum * 255 / pixelCount);
		}
		// paint first row, other cells
		cX += cellWidth/2 - cellWidth;
		cWidth = min(cellWidth, width - x*cellWidth + cellWidth/2);
		cHeight = min(cellHeight / 2, height);
		cellOrigin = src.step * cY + cX;
		cellOffset = src.step - cWidth;
		sp = (uchar *)(src.data + cellOrigin);
		dp = (uchar *)(dst.data + cellOrigin);
		curr = buf.data + (curr - buf.data + 256) % bufSize;
		left = buf.data + (curr - buf.data + bufOffsetLeft) % bufSize;
		for (int b=0; b < cHeight; b++, sp+= cellOffset, dp += cellOffset){
		  for (int a=0; a < cWidth; a++, sp++, dp++){
			  *dp = interp(a,cWidth-a,left[*sp], curr[*sp]);
		  }
		}
	}
	// process (i.e. paint) first row, last cell (only if necessary)
	if (width % cellWidth > cellWidth / 2 || width % cellWidth == 0) {
		cWidth = (width - cellWidth / 2) % cellWidth;
		cHeight = min(cellHeight / 2, height);
		cX = width-cWidth;
		cellOrigin = src.step * cY + cX;
		cellOffset = src.step - cWidth;
		sp = (uchar *)(src.data + cellOrigin);
		dp = (uchar *)(dst.data + cellOrigin);
		for (int b=0; b < cHeight; b++, sp+= cellOffset, dp += cellOffset){
		  for (int a=0; a < cWidth; a++, sp++, dp++){
			*dp = curr[*sp];
		  }
		}
	}
	// process rest of rows
	for (int y = 1; y < gridHeight; y++) {
		// process other rows, first cell
		cX = 0;
		cY = y*cellHeight;
		cWidth = min(cellWidth, width);
		cHeight = min(cellHeight, height - y*cellHeight);
		pixelCount = cWidth*cHeight;
		sum = 0;
		roi.release();
		roi = Mat(src,Rect(cX,cY,cWidth,cHeight));
		hist2u(roi,hist);
		if (clipFactor < 1) clipHistogram(hist,clipFactor,pixelCount);
		// equalization
		if (pbuf == ebuf) pbuf = buf.begin<uchar>();
		for(phist = hist.begin<int>(); phist!=ehist; phist++, pbuf++){
			sum += *phist;
			*pbuf = saturate_cast<uchar>(sum * 255 / pixelCount);
		}
		// paint other rows, first cell
		cY += cellHeight/2 - cellHeight;
		cWidth = min(cellWidth / 2, width);
		cHeight = min(cellHeight, height - y*cellHeight + cellHeight/2);
		cellOrigin = src.step * cY + cX;
		cellOffset = src.step - cWidth;
		sp = (uchar *)(src.data + cellOrigin);
		dp = (uchar *)(dst.data + cellOrigin);
		curr = buf.data + (curr - buf.data + 256) % bufSize;
		top = buf.data + (curr - buf.data + bufOffsetTop) % bufSize;
		for (int b=0; b < cHeight; b++, sp+= cellOffset, dp += cellOffset){
		  for (int a=0; a < cWidth; a++, sp++, dp++){
			  *dp = interp(b,cHeight-b,top[*sp], curr[*sp]);
		  }
		}
		// process other rows, rest of cells
		for (int x = 1; x < gridWidth; x++) {
			cX = x*cellWidth;
			cY = y*cellHeight;
			cWidth = min(cellWidth, width - x*cellWidth);
			cHeight = min(cellHeight, height - y*cellHeight);
			pixelCount = cWidth*cHeight;
			sum = 0;
			roi.release();
			roi = Mat(src,Rect(cX,cY,cWidth,cHeight));
			hist2u(roi,hist);
			if (clipFactor < 1) clipHistogram(hist,clipFactor,pixelCount);
			// equalization
			if (pbuf == ebuf) pbuf = buf.begin<uchar>();
			for(phist = hist.begin<int>(); phist!=ehist; phist++, pbuf++){
				sum += *phist;
				*pbuf = saturate_cast<uchar>(sum * 255 / pixelCount);
			}
			// paint other rows, rest of cells
			cX += cellWidth/2 - cellWidth;
			cY += cellHeight/2 - cellHeight;
			cWidth = min(cellWidth, width - x*cellWidth + cellWidth/2);
			cHeight = min(cellHeight, height - y*cellHeight + cellHeight/2);
			cellOrigin = src.step * cY + cX;
			cellOffset = src.step - cWidth;
			sp = (uchar *)(src.data + cellOrigin);
			dp = (uchar *)(dst.data + cellOrigin);
			curr = buf.data + (curr - buf.data + 256) % bufSize;
			top = buf.data + (curr - buf.data + bufOffsetTop) % bufSize;
			topleft = buf.data + (curr - buf.data + bufOffsetTopLeft) % bufSize;
			left = buf.data + (curr - buf.data + bufOffsetLeft) % bufSize;
			for (int b=0; b < cHeight; b++, sp+= cellOffset, dp += cellOffset){
			  for (int a=0; a < cWidth; a++, sp++, dp++){
				  *dp = interp(a, cWidth-a,b,cHeight-b,topleft[*sp],top[*sp],left[*sp],curr[*sp]);
			  }
			}
		}
		// process (i.e. paint) other rows, last cell (only if necessary)
		if (width % cellWidth > cellWidth / 2 || width % cellWidth == 0) {
			cWidth = (width - cellWidth / 2) % cellWidth;
			cHeight = min(cellHeight, height - y*cellHeight + cellHeight/2);
			cX = width-cWidth;
			cellOrigin = src.step * cY + cX;
			cellOffset = src.step - cWidth;
			sp = (uchar *)(src.data + cellOrigin);
			dp = (uchar *)(dst.data + cellOrigin);
			top = buf.data + (curr - buf.data + bufOffsetTop) % bufSize;
			for (int b=0; b < cHeight; b++, sp+= cellOffset, dp += cellOffset){
			  for (int a=0; a < cWidth; a++, sp++, dp++){
				  *dp = interp(b,cHeight-b,top[*sp], curr[*sp]);
			  }
			}
		}
	}
	// process (i.e. paint) last row (only if necessary)
	if (height % cellHeight > cellHeight / 2 || height % cellHeight == 0) {
		// paint last row, first cell
		cWidth =  min(cellWidth / 2, width);
		cHeight = (height - cellHeight / 2) % cellHeight;
		cX = 0;
		cY = height-cHeight;
		cellOrigin = src.step * cY + cX;
		cellOffset = src.step - cWidth;
		sp = (uchar *)(src.data + cellOrigin);
		dp = (uchar *)(dst.data + cellOrigin);
		curr = buf.data + (curr - buf.data + bufOffsetTop + 256) % bufSize;
		for (int b=0; b < cHeight; b++, sp+= cellOffset, dp += cellOffset){
		  for (int a=0; a < cWidth; a++, sp++, dp++){
			*dp = curr[*sp];
		  }
		}
		// paint last row, other cells
		for (int x = 1; x < gridWidth; x++) {
			cX = (x-1)*cellWidth + cellWidth/2;
			cWidth = min(cellWidth, width - x*cellWidth + cellWidth/2);
			cHeight = (height - cellHeight / 2) % cellHeight;
			cellOrigin = src.step * cY + cX;
			cellOffset = src.step - cWidth;
			sp = (uchar *)(src.data + cellOrigin);
			dp = (uchar *)(dst.data + cellOrigin);
			left = curr;
			curr = buf.data + (curr - buf.data + 256) % bufSize;
			for (int b=0; b < cHeight; b++, sp+= cellOffset, dp += cellOffset){
			  for (int a=0; a < cWidth; a++, sp++, dp++){
				  *dp = interp(a,cWidth-a,left[*sp], curr[*sp]);
			  }
			}
		}
		// paint last row, last cell (only if necessary)
		if (width % cellWidth > cellWidth / 2 || width % cellWidth == 0) {
			cWidth = (width - cellWidth / 2) % cellWidth;
			cHeight = (height - cellHeight / 2) % cellHeight;
			cX = width-cWidth;
			cellOrigin = src.step * cY + cX;
			cellOffset = src.step - cWidth;
			sp = (uchar *)(src.data + cellOrigin);
			dp = (uchar *)(dst.data + cellOrigin);
			for (int b=0; b < cHeight; b++, sp+= cellOffset, dp += cellOffset){
			  for (int a=0; a < cWidth; a++, sp++, dp++){
				  *dp = curr[*sp];
			  }
			}
		}
	}
}


/** ------------------------------- commandline functions ------------------------------- **/

/**
 * Parses a command line
 * This routine should be called for parsing command lines for executables.
 * Note, that all options require '-' as prefix and may contain an arbitrary
 * number of optional arguments.
 *
 * cmd: commandline representation
 * argc: number of parameters
 * argv: string array of argument values
 */
void cmdRead(map<string ,vector<string> >& cmd, int argc, char *argv[]){
	for (int i=1; i< argc; i++){
		char * argument = argv[i];
		if (strlen(argument) > 1 && argument[0] == '-' && (argument[1] < '0' || argument[1] > '9')){
			cmd[argument]; // insert
			char * argument2;
			while (i + 1 < argc && (strlen(argument2 = argv[i+1]) <= 1 || argument2[0] != '-'  || (argument2[1] >= '0' && argument2[1] <= '9'))){
				cmd[argument].push_back(argument2);
				i++;
			}
		}
		else {
			CV_Error(CV_StsBadArg,"Invalid command line format");
		}
	}
}

/**
 * Checks, if each command line option is valid, i.e. exists in the options array
 *
 * cmd: commandline representation
 * validOptions: list of valid options separated by pipe (i.e. |) character
 */
void cmdCheckOpts(map<string ,vector<string> >& cmd, const string validOptions){
	vector<string> tokens;
	const string delimiters = "|";
	string::size_type lastPos = validOptions.find_first_not_of(delimiters,0); // skip delimiters at beginning
	string::size_type pos = validOptions.find_first_of(delimiters, lastPos); // find first non-delimiter
	while (string::npos != pos || string::npos != lastPos){
		tokens.push_back(validOptions.substr(lastPos,pos - lastPos)); // add found token to vector
		lastPos = validOptions.find_first_not_of(delimiters,pos); // skip delimiters
		pos = validOptions.find_first_of(delimiters,lastPos); // find next non-delimiter
	}
	sort(tokens.begin(), tokens.end());
	for (map<string, vector<string> >::iterator it = cmd.begin(); it != cmd.end(); it++){
		if (!binary_search(tokens.begin(),tokens.end(),it->first)){
			CV_Error(CV_StsBadArg,"Command line parameter '" + it->first + "' not allowed.");
			tokens.clear();
			return;
		}
	}
	tokens.clear();
}

/*
 * Checks, if a specific required option exists in the command line
 *
 * cmd: commandline representation
 * option: option name
 */
void cmdCheckOptExists(map<string ,vector<string> >& cmd, const string option){
	map<string, vector<string> >::iterator it = cmd.find(option);
	if (it == cmd.end()) CV_Error(CV_StsBadArg,"Command line parameter '" + option + "' is required, but does not exist.");
}

/*
 * Checks, if a specific option has the appropriate number of parameters
 *
 * cmd: commandline representation
 * option: option name
 * size: appropriate number of parameters for the option
 */
void cmdCheckOptSize(map<string ,vector<string> >& cmd, const string option, const unsigned int size = 1){
	map<string, vector<string> >::iterator it = cmd.find(option);
	if (it->second.size() != size) CV_Error(CV_StsBadArg,"Command line parameter '" + option + "' has unexpected size.");
}

/*
 * Checks, if a specific option has the appropriate number of parameters
 *
 * cmd: commandline representation
 * option: option name
 * min: minimum appropriate number of parameters for the option
 * max: maximum appropriate number of parameters for the option
 */
void cmdCheckOptRange(map<string ,vector<string> >& cmd, string option, unsigned int min = 0, unsigned int max = 1){
	map<string, vector<string> >::iterator it = cmd.find(option);
	unsigned int size = it->second.size();
	if (size < min || size > max) CV_Error(CV_StsBadArg,"Command line parameter '" + option + "' is out of range.");
}

/*
 * Returns the list of parameters for a given option
 *
 * cmd: commandline representation
 * option: name of the option
 */
vector<string> * cmdGetOpt(map<string ,vector<string> >& cmd, const string option){
	map<string, vector<string> >::iterator it = cmd.find(option);
	return (it != cmd.end()) ? &(it->second) : 0;
}

/*
 * Returns number of parameters in an option
 *
 * cmd: commandline representation
 * option: name of the option
 */
unsigned int cmdSizePars(map<string ,vector<string> >& cmd, const string option){
	map<string, vector<string> >::iterator it = cmd.find(option);
	return (it != cmd.end()) ? it->second.size() : 0;
}

/*
 * Returns a specific parameter type (int) given an option and parameter index
 *
 * cmd: commandline representation
 * option: name of option
 * param: name of parameter
 */
int cmdGetParInt(map<string ,vector<string> >& cmd, string option, unsigned int param = 0){
	map<string, vector<string> >::iterator it = cmd.find(option);
	if (it != cmd.end()) {
		if (param < it->second.size()) {
			return atoi(it->second[param].c_str());
		}
	}
	return 0;
}

/*
 * Returns a specific parameter type (float) given an option and parameter index
 *
 * cmd: commandline representation
 * option: name of option
 * param: name of parameter
 */
float cmdGetParFloat(map<string ,vector<string> >& cmd, const string option, const unsigned int param = 0){
	map<string, vector<string> >::iterator it = cmd.find(option);
	if (it != cmd.end()) {
		if (param < it->second.size()) {
			return atof(it->second[param].c_str());
		}
	}
	return 0;
}

/*
 * Returns a specific parameter type (string) given an option and parameter index
 *
 * cmd: commandline representation
 * option: name of option
 * param: name of parameter
 */
string cmdGetPar(map<string ,vector<string> >& cmd, const string option, const unsigned int param = 0){
	map<string, vector<string> >::iterator it = cmd.find(option);
	if (it != cmd.end()) {
		if (param < it->second.size()) {
			return it->second[param];
		}
	}
	return 0;
}

/** ------------------------------- timing functions ------------------------------- **/

/**
 * Class for handling timing progress information
 */
class Timing{
public:
	/** integer indicating progress with respect tot total **/
	int progress;
	/** total count for progress **/
	int total;

	/*
	 * Default constructor for timing initializing time.
	 * Automatically calls init()
	 *
	 * seconds: update interval in seconds
	 * eraseMode: if true, outputs sends erase characters at each print command
	 */
	Timing(long seconds, bool eraseMode){
		updateInterval = seconds;
		progress = 1;
		total = 100;
		eraseCount=0;
		erase = eraseMode;
		init();
	}

	/*
	 * Destructor
	 */
	~Timing(){}

	/*
	 * Initializes timing variables
	 */
	void init(void){
		start = boost::posix_time::microsec_clock::universal_time();
		lastPrint = start - boost::posix_time::seconds(updateInterval);
	}

	/*
	 * Clears printing (for erase option only)
	 */
	void clear(void){
		string erase(eraseCount,'\r');
		erase.append(eraseCount,' ');
		erase.append(eraseCount,'\r');
		printf("%s",erase.c_str());
		eraseCount = 0;
	}

	/*
	 * Updates current time and returns true, if output should be printed
	 */
	bool update(void){
		current = boost::posix_time::microsec_clock::universal_time();
		return ((current - lastPrint > boost::posix_time::seconds(updateInterval)) || (progress == total));
	}

	/*
	 * Prints timing object to STDOUT
	 */
	void print(void){
		lastPrint = current;
		float percent = 100.f * progress / total;
		boost::posix_time::time_duration passed = (current - start);
		boost::posix_time::time_duration togo = passed * (total - progress) / max(1,progress);
		if (erase) {
			string erase(eraseCount,'\r');
			printf("%s",erase.c_str());
			int newEraseCount = (progress != total) ? printf("Progress ... %3.2f%% (%i/%i Total %i:%02i:%02i.%03i Remaining ca. %i:%02i:%02i.%03i)",percent,progress,total,passed.hours(),passed.minutes(),passed.seconds(),(int)(passed.total_milliseconds()%1000),togo.hours(),togo.minutes(),togo.seconds(),(int)(togo.total_milliseconds() % 1000)) : printf("Progress ... %3.2f%% (%i/%i Total %i:%02i:%02i.%03d)",percent,progress,total,passed.hours(),passed.minutes(),passed.seconds(),(int)(passed.total_milliseconds()%1000));
			if (newEraseCount < eraseCount) {
				string erase(newEraseCount-eraseCount,' ');
				erase.append(newEraseCount-eraseCount,'\r');
				printf("%s",erase.c_str());
			}
			eraseCount = newEraseCount;
		}
		else {
			eraseCount = (progress != total) ? printf("Progress ... %3.2f%% (%i/%i Total %i:%02i:%02i.%03i Remaining ca. %i:%02i:%02i.%03i)\n",percent,progress,total,passed.hours(),passed.minutes(),passed.seconds(),(int)(passed.total_milliseconds()%1000),togo.hours(),togo.minutes(),togo.seconds(),(int)(togo.total_milliseconds() % 1000)) : printf("Progress ... %3.2f%% (%i/%i Total %i:%02i:%02i.%03d)\n",percent,progress,total,passed.hours(),passed.minutes(),passed.seconds(),(int)(passed.total_milliseconds()%1000));
		}
	}
private:
	long updateInterval;
	boost::posix_time::ptime start;
	boost::posix_time::ptime current;
	boost::posix_time::ptime lastPrint;
	int eraseCount;
	bool erase;
};

/** ------------------------------- file pattern matching functions ------------------------------- **/


/*
 * Formats a given string, such that it can be used as a regular expression
 * I.e. escapes special characters and uses * and ? as wildcards
 *
 * pattern: regular expression path pattern
 * pos: substring starting index
 * n: substring size
 *
 * returning: escaped substring
 */
string patternSubstrRegex(string& pattern, size_t pos, size_t n){
	string result;
	for (size_t i=pos, e=pos+n; i < e; i++ ) {
		char c = pattern[i];
		if ( c == '\\' || c == '.' || c == '+' || c == '[' || c == '{' || c == '|' || c == '(' || c == ')' || c == '^' || c == '$' || c == '}' || c == ']') {
			result.append(1,'\\');
			result.append(1,c);
		}
		else if (c == '*'){
			result.append("([^/\\\\]*)");
		}
		else if (c == '?'){
			result.append("([^/\\\\])");
		}
		else {
			result.append(1,c);
		}
	}
	return result;
}

/*
 * Converts a regular expression path pattern into a list of files matching with this pattern by replacing wildcards
 * starting in position pos assuming that all prior wildcards have been resolved yielding intermediate directory path.
 * I.e. this function appends the files in the specified path according to yet unresolved pattern by recursive calling.
 *
 * pattern: regular expression path pattern
 * files: the list to which new files can be applied
 * pos: an index such that positions 0...pos-1 of pattern are already considered/matched yielding path
 * path: the current directory (or empty)
 */
void patternToFiles(string& pattern, vector<string>& files, const size_t& pos, const string& path){
	size_t first_unknown = pattern.find_first_of("*?",pos); // find unknown * in pattern
	if (first_unknown != string::npos){
		size_t last_dirpath = pattern.find_last_of("/\\",first_unknown);
		size_t next_dirpath = pattern.find_first_of("/\\",first_unknown);
		if (next_dirpath != string::npos){
			boost::regex expr((last_dirpath != string::npos && last_dirpath > pos) ? patternSubstrRegex(pattern,last_dirpath+1,next_dirpath-last_dirpath-1) : patternSubstrRegex(pattern,pos,next_dirpath-pos));
			boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end
			try {
				for ( boost::filesystem::directory_iterator itr( ((path.length() > 0) ? path + pattern[pos-1] : (last_dirpath != string::npos && last_dirpath > pos) ? "" : "./") + ((last_dirpath != string::npos && last_dirpath > pos) ? pattern.substr(pos,last_dirpath-pos) : "")); itr != end_itr; ++itr )
				{
					if (boost::filesystem::is_directory(itr->path())){
						boost::filesystem::path p = itr->path().filename();
						string s =  p.string();
						if (boost::regex_match(s.c_str(), expr)){
							patternToFiles(pattern,files,(int)(next_dirpath+1),((path.length() > 0) ? path + pattern[pos-1] : "") + ((last_dirpath != string::npos && last_dirpath > pos) ? pattern.substr(pos,last_dirpath-pos) + pattern[last_dirpath] : "") + s);
						}
					}
				}
			}
			catch (boost::filesystem::filesystem_error &e){}
		}
		else {
			boost::regex expr((last_dirpath != string::npos && last_dirpath > pos) ? patternSubstrRegex(pattern,last_dirpath+1,pattern.length()-last_dirpath-1) : patternSubstrRegex(pattern,pos,pattern.length()-pos));
			boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end
			try {
				for ( boost::filesystem::directory_iterator itr(((path.length() > 0) ? path +  pattern[pos-1] : (last_dirpath != string::npos && last_dirpath > pos) ? "" : "./") + ((last_dirpath != string::npos && last_dirpath > pos) ? pattern.substr(pos,last_dirpath-pos) : "")); itr != end_itr; ++itr )
				{
					boost::filesystem::path p = itr->path().filename();
					string s =  p.string();
					if (boost::regex_match(s.c_str(), expr)){
						files.push_back(((path.length() > 0) ? path + pattern[pos-1] : "") + ((last_dirpath != string::npos && last_dirpath > pos) ? pattern.substr(pos,last_dirpath-pos) + pattern[last_dirpath] : "") + s);
					}
				}
			}
			catch (boost::filesystem::filesystem_error &e){}
		}
	}
	else { // no unknown symbols
		boost::filesystem::path file(((path.length() > 0) ? path + "/" : "") + pattern.substr(pos,pattern.length()-pos));
		if (boost::filesystem::exists(file)){
			files.push_back(file.string());
		}
	}
}

/**
 * Converts a regular expression path pattern into a list of files matching with this pattern
 *
 * pattern: regular expression path pattern
 * files: the list to which new files can be applied
 */
void patternToFiles(string& pattern, vector<string>& files){
	patternToFiles(pattern,files,0,"");
}

/*
 * Renames a given filename corresponding to the actual file pattern using a renaming pattern.
 * Wildcards can be referred to as ?1, ?2, ... in the order they appeared in the file pattern.
 *
 * pattern: regular expression path pattern
 * renamePattern: renaming pattern using ?1, ?2, ... as placeholders for wildcards
 * infile: path of the file (matching with pattern) to be renamed
 * outfile: path of the renamed file
 * par: used parameter (default: '?')
 */
void patternFileRename(string& pattern, const string& renamePattern, const string& infile, string& outfile, const char par = '?'){
	size_t first_unknown = renamePattern.find_first_of(par,0); // find unknown ? in renamePattern
	if (first_unknown != string::npos){
		string formatOut = "";
		for (size_t i=0, e=renamePattern.length(); i < e; i++ ) {
			char c = renamePattern[i];
			if ( c == par && i+1 < e) {
				c = renamePattern[i+1];
				if (c > '0' && c <= '9'){
					formatOut.append(1,'$');
					formatOut.append(1,c);
				}
				else {
					formatOut.append(1,par);
					formatOut.append(1,c);
				}
				i++;
			}
			else {
				formatOut.append(1,c);
			}
		}
		boost::regex patternOut(patternSubstrRegex(pattern,0,pattern.length()));
		outfile = boost::regex_replace(infile,patternOut,formatOut,boost::match_default | boost::format_perl);
	} else {
		outfile = renamePattern;
	}
}

void polyfit(const Mat& src_x, const Mat& src_y, Mat& dst, int order)
{
	    CV_Assert((src_x.rows>0)&&(src_y.rows>0)&&(src_x.cols==1)&&(src_y.cols==1)
	            &&(dst.cols==1)&&(dst.rows==(order+1))&&(order>=1));
	    Mat X;
	    X = Mat::zeros(src_x.rows, order+1,CV_32FC1);
	    Mat copy;
	    for(int i = 0; i <=order;i++)
	    {
	        copy = src_x.clone();
	        pow(copy,i,copy);
	        Mat M1 = X.col(i);
	        copy.col(0).copyTo(M1);
	    }
	    Mat X_t, X_inv;
	    transpose(X,X_t);
	    Mat temp = X_t*X;
	    Mat temp2;
	    invert (temp,temp2);
	    Mat temp3 = temp2*X_t;
	    Mat W = temp3*src_y;
	    W.copyTo(dst);
}

// samples a polynom
void polysample(const Mat& poly, vector<Point2f>& curve, const double xStart, const double xEnd, const int samples){
	CV_Assert(poly.cols == 1 && poly.rows == 3 && poly.type() == CV_32FC1);
	float * polyData = (float*) poly.data;
	for (int i=0; i<samples; i++){
		double x = xStart + (i*(xEnd-xStart))/(samples-1);
		curve.push_back(Point2f(x,polyData[0]+polyData[1]*x+polyData[2]*x*x));
	}
}



void loadPoints(string filename, vector<Point2f>& points) {
	ifstream coordfile;
	coordfile.open(filename.c_str(),ios::in);
	//CV_Assert(coordfile.is_open());
	while (!coordfile.eof()){
		string line;
		vector<string> coords;
		getline (coordfile,line);
		if (!line.empty()){
			boost::split(coords,line,boost::is_any_of("\t "));
			CV_Assert(coords.size() == 2);
			points.push_back(Point(atof(coords[0].c_str()),atof(coords[1].c_str())));
		}
	}
	coordfile.close();
}

void loadPolyPoints(string filename, vector<Point2f>& points, int width){
	vector<Point2f> points2;
	loadPoints(filename,points2);
	if (points2.size() >= 3) {
		Mat res(3,1,CV_32FC1);
		Mat x(points2.size(),1,CV_32FC1);
		Mat y(points2.size(),1,CV_32FC1);
		float *xd = (float *) x.data;
		float *yd = (float *) y.data;
		for (vector<Point2f>::iterator it = points2.begin(); it != points2.end(); ++it,++xd,++yd){
			*xd = (*it).x;
			*yd = (*it).y;
		}
		polyfit(x,y,res,2);
		polysample(res, points, 0, width-1, width);
	}
}

/**
 * Samples an ellipse in cartesian coordinates
 * cart: CV_32FC2 1 x size cartesian coordinates
 * ellipse: ellipse to be sampled
 */
void ellipse2Cart(Mat& cart, const RotatedRect& ellipse) {
	CV_Assert(cart.type() == CV_32FC2);
	float centerX = ellipse.center.x;
	float centerY = ellipse.center.y;
	float ellA = ellipse.size.width/2;
	float ellB = ellipse.size.height/2;
	double alpha = (ellipse.angle)*M_PI/180; // inclination angle of the ellipse
	if (alpha > M_PI) alpha -= M_PI; // normalize alpha to [0,M_PI]
	const double cosAlpha = cos(alpha);
	const double sinAlpha = sin(alpha);
	float * d = (float *)cart.data;
	int width = cart.cols;
	const float thetaoffset = 2 * M_PI / cart.cols;
	float theta = 0;
	for (int x=0; x<width; x++, d++, theta += thetaoffset){
		float beta = (alpha <= theta) ? theta - alpha : 2 * M_PI + theta - alpha; // angle of polar ray in ellipse coords (alpha + beta = theta)
		float s = ellA * cos(beta), t = ellB * sin(beta);
		*d = centerX + cosAlpha * s - sinAlpha * t; // x coordiante
		d++;
		*d = centerY + sinAlpha * s + cosAlpha * t; // y coordinate
	}
}

/** ------------------------------- Rubbersheet transform ------------------------------- **/

/*
 * interpolation mode for rubbersheet repeating the last pixel for a given angle if no values are available
 * (otherwise behaves like INTER_LINEAR)
 */
static const int INTER_LINEAR_REPEAT = 82;

/*
 * Calculates the mapped (polar) image of source using two transformation contours.
 *
 * src: CV_8U (cartesian) source image (possibly multi-channel)
 * dst:	CV_8U (polar) destination image (possibly multi-channel, 2 times the col size of inner)
 * inner: CV_32FC2 inner cartesian coordinates
 * outer: CV_32FC2 outer cartesian coordinates
 * interpolation: interpolation mode (INTER_NEAREST, INTER_LINEAR or INTER_LINEAR_REPEAT)
 * fill: fill value for pixels out of the image
 */
void rubbersheet(const Mat& src, Mat& dst, const Mat& inner, const Mat& outer, const int interpolation = INTER_LINEAR, const uchar fill = 0) {
	int nChannels = src.channels();
	int dstheight = dst.rows;
	int dstwidth = dst.cols;
	int srcheight = src.rows;
	int srcwidth = src.cols;
	uchar * pdst = dst.data;
	int dstoffset = dst.step - dstwidth * nChannels;
	int srcstep = src.step;
	float roffset = 1.f / dstheight;
	float r = 0;
	if (interpolation == INTER_NEAREST){
		for (int y=0; y < dstheight; y++, pdst+= dstoffset, r+=roffset){
			float * pinner = (float *) inner.data;
			float * pouter = (float *) outer.data;
			for (int x=0; x < dstwidth; x++, pinner++, pouter++){
				float a = *pinner + r * (*pouter - *pinner);
				pinner++; pouter++;
				float b =  *pinner + r * (*pouter - *pinner);
				int coordX = cvRound(a);
				int coordY = cvRound(b);
				if (coordX < 0 || coordY < 0 || coordX >= srcwidth || coordY >= srcheight){
					for (int i=0; i< nChannels; i++,pdst++){
						*pdst = fill;
					}
				}
				else {
					uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
					for (int i=0; i< nChannels; i++,pdst++,psrc++){
						*pdst = *psrc;
					}
				}
			}
		}
	}
	else if (interpolation == INTER_LINEAR){
		for (int y=0; y < dstheight; y++, pdst+= dstoffset, r+= roffset){
			float * pinner = (float *) inner.data;
			float * pouter = (float *) outer.data;
			for (int x=0; x < dstwidth; x++, pinner++, pouter++){
				float a = *pinner + r * (*pouter - *pinner);
				pinner++; pouter++;
				float b =  *pinner + r * (*pouter - *pinner);
				int coordX = cvFloor(a);
				int coordY = cvFloor(b);
				if (coordX >= 0){
					if (coordY >= 0){
						if (coordX < srcwidth-1){
							if (coordY < srcheight-1){
								float dx = a-coordX;
								float dy = b-coordY;
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = saturate_cast<uchar>((1-dy)*((1-dx)*((float)(*psrc)) + dx*(float)(psrc[nChannels]))+ dy*((1-dx)*((float)(psrc[srcstep])) + dx*(float)(psrc[nChannels+srcstep])));
								}
							}
							else if (coordY == srcheight-1){ // bottom out
								float dx = a-coordX;
								float dy = b-coordY;
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = saturate_cast<uchar>((1-dy)*((1-dx)*((float)(*psrc)) + dx*(float)(psrc[nChannels])) + dy * ((float) fill));
								}
							}
							else {
								for (int i=0; i< nChannels; i++,pdst++){
									*pdst = fill;
								}
							}
						}
						else if (coordX == srcwidth-1){
							if (coordY < srcheight-1){// right out
								float dx = a-coordX;
								float dy = b-coordY;
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = saturate_cast<uchar>((1-dx)*((1-dy)*((float)(*psrc))+ dy*((float)(psrc[srcstep]))) + dx * ((float) fill));
								}
							}
							else if (coordY == srcheight-1){ // bottom right out
								float dx = a-coordX;
								float dy = b-coordY;
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = saturate_cast<uchar>((1-dy)*((1-dx)*((float)(*psrc)) + dx * ((float)fill)) + dy * ((float) fill));
								}
							}
							else {
								for (int i=0; i< nChannels; i++,pdst++){
									*pdst = fill;
								}
							}
						}
						else {
							for (int i=0; i< nChannels; i++,pdst++){
								*pdst = fill;
							}
						}
					}
					else if (coordY == -1){
						if (coordX < srcwidth-1){// top out
								float dx = a-coordX;
								float dy = b-coordY;
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = saturate_cast<uchar>((dy)*((1-dx)*((float)(psrc[srcstep])) + dx*(float)(psrc[nChannels+srcstep])) + (1-dy) * ((float) fill));
								}
						}
						else if (coordX == srcwidth-1){// top right out
								float dx = a-coordX;
								float dy = b-coordY;
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = saturate_cast<uchar>((dy)*((1-dx)*((float)(psrc[srcstep])) + dx * ((float)fill)) + (1-dy) * ((float) fill));
								}
						}
						else {
							for (int i=0; i< nChannels; i++,pdst++){
								*pdst = fill;
							}
						}
					}
					else {
						for (int i=0; i< nChannels; i++,pdst++){
							*pdst = fill;
						}
					}
				}
				else if (coordX == -1){
					if (coordY >= 0){
						if (coordY < srcheight-1){// left out
							float dx = a-coordX;
							float dy = b-coordY;
							uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
							for (int i=0; i< nChannels; i++,pdst++,psrc++){
								*pdst = saturate_cast<uchar>(dx*((1-dy)*(float)(psrc[nChannels]) + dy*(float)(psrc[nChannels+srcstep])) + (1-dx) * ((float) fill));
							}
						}
						else if (coordY == srcheight-1){ // left bottom out
							float dx = a-coordX;
							float dy = b-coordY;
							uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
							for (int i=0; i< nChannels; i++,pdst++,psrc++){
								*pdst = saturate_cast<uchar>((1-dy)*((dx)*((float)(psrc[nChannels])) + (1-dx) * ((float) fill)) + dy * ((float) fill));
							}
						}
						else {
							for (int i=0; i< nChannels; i++,pdst++){
								*pdst = fill;
							}
						}
					}
					else if (coordY == -1){ // left top out
						float dx = a-coordX;
						float dy = b-coordY;
						uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
						for (int i=0; i< nChannels; i++,pdst++,psrc++){
							*pdst = saturate_cast<uchar>((dy)*((dx)*((float)(psrc[nChannels+srcstep])) + (1-dx) * ((float) fill)) + (1-dy) * ((float) fill));
						}
					}
					else {
						for (int i=0; i< nChannels; i++,pdst++){
							*pdst = fill;
						}
					}
				}
				else {
					for (int i=0; i< nChannels; i++,pdst++){
						*pdst = fill;
					}
				}
			}
		}
	}
	else { // INTER_LINEAR_REPEAT (repeats last pixel value)
		uchar * firstLine = pdst + dst.step;
		int step = dst.step;
		for (int y=0; y < dstheight; y++, pdst+= dstoffset, r+= roffset){
			float * pinner = (float *) inner.data;
			float * pouter = (float *) outer.data;
			for (int x=0; x < dstwidth; x++, pinner++, pouter++){
				float a = *pinner + r * (*pouter - *pinner);
				pinner++; pouter++;
				float b =  *pinner + r * (*pouter - *pinner);
				int coordX = cvFloor(a);
				int coordY = cvFloor(b);
				if (coordX >= 0){
					if (coordY >= 0){
						if (coordX < srcwidth-1){
							if (coordY < srcheight-1){
								float dx = a-coordX;
								float dy = b-coordY;
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = saturate_cast<uchar>((1-dy)*((1-dx)*((float)(*psrc)) + dx*(float)(psrc[nChannels]))+ dy*((1-dx)*((float)(psrc[srcstep])) + dx*(float)(psrc[nChannels+srcstep])));
								}
							}
							else if (coordY == srcheight-1){ // bottom out
								float dx = a-coordX;
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = saturate_cast<uchar>(((1-dx)*((float)(*psrc)) + dx*(float)(psrc[nChannels])));
								}
							}
							else if (pdst >= firstLine){
								for (int i=0; i< nChannels; i++,pdst++){
									*pdst = *(pdst - step); // one row above
								}
							}
							else {
								for (int i=0; i< nChannels; i++,pdst++){
									*pdst = fill;
								}
							}
						}
						else if (coordX == srcwidth-1){
							if (coordY < srcheight-1){// right out
								float dy = b-coordY;
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = saturate_cast<uchar>(((1-dy)*((float)(*psrc))+ dy*((float)(psrc[srcstep]))));
								}
							}
							else if (coordY == srcheight-1){ // bottom right out
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = *psrc;
								}
							}
							else if (pdst >= firstLine){
								for (int i=0; i< nChannels; i++,pdst++){
									*pdst = *(pdst - step); // one row above
								}
							}
							else {
								for (int i=0; i< nChannels; i++,pdst++){
									*pdst = fill;
								}
							}
						}
						else {
							for (int i=0; i< nChannels; i++,pdst++){
								*pdst = *(pdst - step); // one row above
							}
						}
					}
					else if (coordY == -1){
						if (coordX < srcwidth-1){// top out
								float dx = a-coordX;
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = saturate_cast<uchar>(((1-dx)*((float)(psrc[srcstep])) + dx*(float)(psrc[nChannels+srcstep])));
								}
						}
						else if (coordX == srcwidth-1){// top right out
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = psrc[srcstep];
								}
						}
						else if (pdst >= firstLine){
							for (int i=0; i< nChannels; i++,pdst++){
								*pdst = *(pdst - step); // one row above
							}
						}
						else {
							for (int i=0; i< nChannels; i++,pdst++){
								*pdst = fill;
							}
						}
					}
					else if (pdst >= firstLine){
						for (int i=0; i< nChannels; i++,pdst++){
							*pdst = *(pdst - step); // one row above
						}
					}
					else {
						for (int i=0; i< nChannels; i++,pdst++){
							*pdst = fill;
						}
					}
				}
				else if (coordX == -1){
					if (coordY >= 0){
						if (coordY < srcheight-1){// left out
							float dy = b-coordY;
							uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
							for (int i=0; i< nChannels; i++,pdst++,psrc++){
								*pdst = saturate_cast<uchar>(((1-dy)*(float)(psrc[nChannels]) + dy*(float)(psrc[nChannels+srcstep])));
							}
						}
						else if (coordY == srcheight-1){ // bottom left out
							uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
							for (int i=0; i< nChannels; i++,pdst++,psrc++){
								*pdst = psrc[nChannels];
							}
						}
						else if (pdst >= firstLine){
							for (int i=0; i< nChannels; i++,pdst++){
								*pdst = *(pdst - step); // one row above
							}
						}
						else {
							for (int i=0; i< nChannels; i++,pdst++){
								*pdst = fill;
							}
						}
					}
					else if (coordY == -1){ // top left out
						uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
						for (int i=0; i< nChannels; i++,pdst++,psrc++){
							*pdst = psrc[nChannels+srcstep];
						}
					}
					else if (pdst >= firstLine){
						for (int i=0; i< nChannels; i++,pdst++){
							*pdst = *(pdst - step); // one row above
						}
					}
					else {
						for (int i=0; i< nChannels; i++,pdst++){
							*pdst = fill;
						}
					}
				}
				else if (pdst >= firstLine){
					for (int i=0; i< nChannels; i++,pdst++){
						*pdst = *(pdst - step); // one row above
					}
				}
				else {
					for (int i=0; i< nChannels; i++,pdst++){
						*pdst = fill;
					}
				}
			}
		}
	}
}


void reverse(vector<Point2f>& curve,vector<Point2f>& reversed){
	for (vector<Point2f>::iterator it = curve.end(); it != curve.begin(); --it){
		reversed.push_back(*(it-1));
	}
}

/*
 * Main program
 */
int main(int argc, char *argv[])
{
	int mode = MODE_HELP;
	map<string,vector<string> > cmd;
	try {
		cmdRead(cmd,argc,argv);
		if (cmd.size() == 0 || cmdGetOpt(cmd,"-h") != 0) mode = MODE_HELP;
    	else mode = MODE_MAIN;
    	if (mode == MODE_MAIN) {
			// validate command line
			cmdCheckOpts(cmd,"-i|-c|-o|-m|-s|-sr|-srl|-roi|-e|-q|-t");
			cmdCheckOptExists(cmd,"-i");
			cmdCheckOptSize(cmd,"-i",1);
			string infiles = cmdGetPar(cmd,"-i",0);
			cmdCheckOptExists(cmd,"-c");
			cmdCheckOptSize(cmd,"-c",4);
			string innerfiles = cmdGetPar(cmd,"-c",0);
			string outerfiles = cmdGetPar(cmd,"-c",1);
			string upperfiles = cmdGetPar(cmd,"-c",2);
			string lowerfiles = cmdGetPar(cmd,"-c",3);
			cmdCheckOptExists(cmd,"-o");
			cmdCheckOptSize(cmd,"-o",1);
			string outfiles = cmdGetPar(cmd,"-o");
			string maskfiles;
			if (cmdGetOpt(cmd,"-m") != 0){
				cmdCheckOptSize(cmd,"-m",1);
				maskfiles = cmdGetPar(cmd,"-m");
			}
			string segresfiles;
			if (cmdGetOpt(cmd,"-sr") != 0){
				cmdCheckOptSize(cmd,"-sr",1);
				segresfiles = cmdGetPar(cmd,"-sr");
			}
			string segreslfiles;
			if (cmdGetOpt(cmd,"-srl") != 0){
				cmdCheckOptSize(cmd,"-srl",1);
				segreslfiles = cmdGetPar(cmd,"-srl");
			}
			string roifiles;
			if (cmdGetOpt(cmd,"-roi") != 0){
				cmdCheckOptSize(cmd,"-roi",1);
				roifiles = cmdGetPar(cmd,"-roi");
			}
			int outWidth = 512, outHeight = 64;
			if (cmdGetOpt(cmd,"-s") != 0){
				cmdCheckOptSize(cmd,"-s",2);
				outWidth = cmdGetParInt(cmd,"-s",0);
				outHeight = cmdGetParInt(cmd,"-s",1);
			}
			bool enhance = false;
			if (cmdGetOpt(cmd,"-e") != 0){
				cmdCheckOptSize(cmd,"-e",0);
				enhance = true;
			}
			bool q = false;
			if (cmdGetOpt(cmd,"-q") != 0){
				cmdCheckOptSize(cmd,"-q",0);
				q = true;
			}
			bool t = false;
			if (cmdGetOpt(cmd,"-t") != 0){
				cmdCheckOptSize(cmd,"-t",0);
				t = true;
			}
			// starting routine
			Timing timing(1,q);
			vector<string> files;
		    patternToFiles(infiles,files);
			CV_Assert(files.size() > 0);
			timing.total = files.size();
			for (vector<string>::iterator infile = files.begin(); infile != files.end(); ++infile, timing.progress++){
				if (!q) cout << "Loading image '" << *infile << "' ..."<< endl;
				Mat img = imread(*infile, CV_LOAD_IMAGE_GRAYSCALE);
				CV_Assert(img.data != 0);
				CV_Assert(img.channels() == 1 || img.channels() == 3);
				CV_Assert(img.depth() == CV_8U);
				string innerfile, outerfile, upperfile, lowerfile;
				patternFileRename(infiles,innerfiles,*infile,innerfile);
				patternFileRename(infiles,outerfiles,*infile,outerfile);
				patternFileRename(infiles,upperfiles,*infile,upperfile);
				patternFileRename(infiles,lowerfiles,*infile,lowerfile);
				vector<Point2f> innerPoints, outerPoints, upperPoints, lowerPoints, rlowerPoints;
				if (!q) cout << "done" << endl << "Loading coordinates '" << innerfile << "' ... ";
				loadPoints(innerfile,innerPoints);
				if (!q) cout << innerPoints.size() << " Points found " << endl;
				if (!q) cout << "Loading coordinates '" << outerfile << "' ... ";
				loadPoints(outerfile,outerPoints);
				if (!q) cout << outerPoints.size() << " Points found " << endl;
				if (!q) cout << "Loading coordinates '" << upperfile << "' ... ";
				loadPolyPoints(upperfile,upperPoints,img.cols);
				if (!q) cout << upperPoints.size() << " Points found " << endl;
				upperPoints.push_back(Point2f(img.cols+1,0.f));
				upperPoints.push_back(Point2f(0.f,0.f));
				if (!q) cout << "Loading coordinates '" << lowerfile << "' ... ";
				loadPolyPoints(lowerfile,lowerPoints,img.cols);
				if (!q) cout << lowerPoints.size() << " Points found " << endl;
				reverse(lowerPoints,rlowerPoints);
				rlowerPoints.push_back(Point2f(0.f,img.rows));
				rlowerPoints.push_back(Point2f(img.cols,img.rows));
				Mat bw(img.rows,img.cols,CV_8UC1);
				Mat innerBoundary(1,outWidth,CV_32FC2);
				Mat outerBoundary(1,outWidth,CV_32FC2);
				bw.setTo(Scalar(0,0,0));
				RotatedRect inner = fitEllipse(innerPoints);
				ellipse2Cart(innerBoundary,inner);
				RotatedRect outer = fitEllipse(outerPoints);
				ellipse2Cart(outerBoundary,outer);
				Mat out (outHeight,outWidth,CV_8UC1);
				rubbersheet(img, out, innerBoundary, outerBoundary, INTER_LINEAR);
				if (enhance){
					int width = img.cols;
					int height = img.rows;
					if (!q) printf("Enhancing texture ...\n");
					clahe(out,out,width/8,height/2);
				}
				if (!maskfiles.empty() || !segresfiles.empty()|| !segreslfiles.empty()){
					ellipse(bw,outer,Scalar(255,255,255),-1);
					ellipse(bw,inner,Scalar(0,0,0),-1);
					int nupper = (int)upperPoints.size();
					Point upper_points[1][nupper];
					for (int i=0; i<nupper; i++){
						upper_points[0][i] = upperPoints[i];
					}

					const Point* upPoints[1] = { upper_points[0] };

					int nlower = (int)rlowerPoints.size();
					Point rlower_points[1][nupper];
					for (int i=0; i<nlower; i++){
						rlower_points[0][i] = rlowerPoints[i];
					}
					const Point* downPoints[1] = { rlower_points[0] };
					if (!segreslfiles.empty()){
						string segreslfile;
						patternFileRename(infiles,segreslfiles,*infile,segreslfile);
						if (!q) cout << "Storing image '" << segreslfile << "' ..."<< endl;
						if (!imwrite(segreslfile,bw)) CV_Error(CV_StsError,"Could not save image '" + segreslfile + "'");
						if (!q) cout << "done" << endl;
					}
					fillPoly(bw,upPoints,&nupper,1,Scalar(0,0,0));
					fillPoly(bw,downPoints,&nlower,1,Scalar(0,0,0));
					if (!segresfiles.empty()){
						string segresfile;
						patternFileRename(infiles,segresfiles,*infile,segresfile);
						if (!q) cout << "Storing image '" << segresfile << "' ..."<< endl;
						if (!imwrite(segresfile,bw)) CV_Error(CV_StsError,"Could not save image '" + segresfile + "'");
						if (!q) cout << "done" << endl;
					}

					if (!maskfiles.empty()){
						Mat outNoise (outHeight,outWidth,CV_8UC1);
						rubbersheet(bw, outNoise, innerBoundary, outerBoundary, INTER_NEAREST);
						string maskfile;
						patternFileRename(infiles,maskfiles,*infile,maskfile);
						if (!q) cout << "Storing image '" << maskfile << "' ..."<< endl;

						if (!imwrite(maskfile,outNoise)) CV_Error(CV_StsError,"Could not save image '" + maskfile + "'");
						if (!q) cout << "done" << endl;
					}
				}
				if (!roifiles.empty()){
					Mat roi(img.rows,img.cols,CV_8UC1);
					roi.setTo(230);
					ellipse(roi,outer,Scalar(255,255,255),-1);
					ellipse(roi,inner,Scalar(0,0,0),-1);
					int nupper = (int)upperPoints.size();
					Point upper_points[1][nupper];
					for (int i=0; i<nupper; i++){
						upper_points[0][i] = upperPoints[i];
					}

					const Point* upPoints[1] = { upper_points[0] };

					int nlower = (int)rlowerPoints.size();
					Point rlower_points[1][nupper];
					for (int i=0; i<nlower; i++){
						rlower_points[0][i] = rlowerPoints[i];
					}
					const Point* downPoints[1] = { rlower_points[0] };

					fillPoly(roi,upPoints,&nupper,1,Scalar(128));
					fillPoly(roi,downPoints,&nlower,1,Scalar(128));
					MatConstIterator_<uchar> isrc = img.begin<uchar>();
					MatIterator_<uchar> idst = roi.begin<uchar>();
					MatConstIterator_<uchar> iend = img.end<uchar>();
					for(;isrc!=iend;isrc++,idst++){
						if (*idst == 255) *idst = *isrc;
					}
					string roifile;
					patternFileRename(infiles,roifiles,*infile,roifile);
					if (!q) cout << "Storing image '" << roifile << "' ..."<< endl;
					if (!imwrite(roifile,roi)) CV_Error(CV_StsError,"Could not save image '" + roifile + "'");
					if (!q) cout << "done" << endl;

				}
				/*
				if (ellpoints.size() >= 5) {
					if (!q) cout << "done" << endl << "Fitting ellipse ..."<< endl;
					RotatedRect ell = fitEllipse(ellpoints);
					ellipse(img,ell,Scalar(lb,lg,lr),thickness);
				}
				if (!q) cout << "done" << endl << "Drawing points ..."<< endl;
				if (ellpoints.size() > 0) {
					for(vector<Point2f>::iterator p = ellpoints.begin(); p < ellpoints.end(); ++p){
						line(img,Point((*p).x,(*p).y-2*thickness),Point((*p).x,(*p).y+2*thickness),Scalar(pb,pg,pr),thickness);
						line(img,Point((*p).x-2*thickness,(*p).y),Point((*p).x+2*thickness,(*p).y),Scalar(pb,pg,pr),thickness);
					}
				}
				*/
				string outfile;
				patternFileRename(infiles,outfiles,*infile,outfile);
				if (!q) cout << "done" << endl << "Storing image '" << outfile << "' ..."<< endl;
				if (!imwrite(outfile,out)) CV_Error(CV_StsError,"Could not save image '" + outfile + "'");
				if (!q) cout << "done" << endl;
				if (t && timing.update()) timing.print();
			}
			if (t && q) timing.clear();
    	}
    	else if (mode == MODE_HELP){
			// validate command line
			cmdCheckOpts(cmd,"-h");
			if (cmdGetOpt(cmd,"-h") != 0) cmdCheckOptSize(cmd,"-h",0);
			// starting routine
			printUsage();
    	}
    }
	catch (...){
		cerr << "Exit with errors." << endl;
		exit(EXIT_FAILURE);
	}
    return EXIT_SUCCESS;
}
