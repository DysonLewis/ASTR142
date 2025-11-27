#include <stdio.h>  // Preprocessor directive to include the standard input/output library

// Function prototype
void printMessage();

int main() {  // Main function where execution starts
    printf("Hello, World!\n");  // Printing text
    printMessage();  // Calling user-defined function
    return 0;  // Returning 0 to indicate successful execution
}

// Function definition
void printMessage() {
    printf("This is a custom function.\n");
}

