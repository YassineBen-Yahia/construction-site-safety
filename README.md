## Construction Site Safety

## Project Overview
The Construction Site Safety project is designed to enhance the safety protocols and procedures on construction sites through effective training, monitoring, and reporting. Utilizing Python for backend processing, Jupyter Notebooks for interactive data analysis, and Docker for containerization, this project aims to provide a comprehensive solution for managing construction site safety efficiently.

## Features
- **Interactive Training Sessions:** Conduct engaging training sessions for site personnel using Jupyter Notebooks.
- **Real-time Monitoring:** Utilize data collected from various sensors to monitor safety conditions on-site in real-time.
- **Automated Reporting:** Generate safety compliance reports automatically based on real-time data analytics.
- **Docker Integration:** Deploy applications in isolated containers to ensure consistency across different environments.

## Installation
To set up the Construction Site Safety project locally, follow the steps below:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/YassineBen-Yahia/construction-site-safety.git
   cd construction-site-safety
   ```
2. **Install Docker:**
   Ensure that Docker is installed on your machine. Follow the [installation guide](https://docs.docker.com/get-docker/) for your operating system.
3. **Build the Docker Image:**
   ```bash
   docker build -t construction-site-safety .
   ```
4. **Run the Application:**
   ```bash
   docker run -p 8888:8888 construction-site-safety
   ```

## Usage
Once the application is running, you can access it through your web browser at `http://localhost:8888`. Here, you can explore the interactive training materials and utilize the monitoring tools available in the application.

## Contributing
We welcome contributions to enhance the Construction Site Safety project. To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with clear messages.
4. Push your changes to your forked repository.
5. Create a pull request detailing your changes.

Together, we can create a safer environment for construction workers.
