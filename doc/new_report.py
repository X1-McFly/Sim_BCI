import os

def main() -> None:
    date = input("Enter Date: ")
    filename = f'report({date})'
    os.mkdir(filename)
    os.chdir(filename)

    with open(f'{filename}.md', "w") as file:
        file.write("# **The Quron Project Report _**\n")
        file.write("<script id=\"MathJax-script\" async src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js\"></script>\n")
        file.write("**Author**: Martin McCorkle <br> **Email**: mamccorkle1@ualr.edu <br> **Date**: December 0, 2024 <br>\n")

        file.write("\n## **1. Project Goal**\n")
        file.write("With industry-leading 8- and 16-channel EEG monitoring devices retailing for upwards of \$2,000 and \$3,500, respectively, there is a growing need for affordable, entry-level alternatives as the BCI market continues to expand. The Quron Project builds on well-documented hardware and aims to provide a cost-effective, open-source solution, making neuroscience education more accessible for individual makers, students, and university programs. Beyond education, the project also seeks to develop a commercial option that balances affordability with high performance, while driving innovative applications for BCIs, such as mind-controlled systems in automotive technology, immersive VR/XR platforms, and neurorehabilitation tools.")

    os.mkdir("img")

if __name__ == '__main__':
    main()