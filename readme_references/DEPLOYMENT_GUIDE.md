# Deployment Guide

Complete guide for deploying the Customer Churn Prediction System in various environments.

## Table of Contents
1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Production Checklist](#production-checklist)

---

## Local Development

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment tool

### Quick Start (Recommended)

**On Windows:**
```bash
quickstart.bat
```

**On Mac/Linux:**
```bash
chmod +x quickstart.sh
./quickstart.sh
```

### Manual Setup

1. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Initialize Database**
```bash
python -c "from database import init_database; init_database()"
```

4. **Run Application**
```bash
streamlit run app.py
```

5. **Access Application**
Open browser and navigate to: `http://localhost:8501`

---

## Docker Deployment

### Prerequisites
- Docker installed
- Docker Compose (optional)

### Using Docker Directly

**Build Image**
```bash
docker build -t churn-prediction:latest .
```

**Run Container**
```bash
docker run -p 8501:8501 \
  -v $(pwd)/database:/app/database \
  -v $(pwd)/datasets:/app/datasets \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  churn-prediction:latest
```

### Using Docker Compose (Recommended)

**Start Application**
```bash
docker-compose up -d
```

**View Logs**
```bash
docker-compose logs -f app
```

**Stop Application**
```bash
docker-compose down
```

### Docker Environment Variables

Create `.env` file:
```
DEBUG=False
LOG_LEVEL=INFO
STREAMLIT_SERVER_HEADLESS=true
```

---

## Cloud Deployment

### Streamlit Cloud (Simplest)

1. **Push to GitHub**
```bash
git add .
git commit -m "Deploy to Streamlit Cloud"
git push
```

2. **Deploy on Streamlit Cloud**
- Go to [share.streamlit.io](https://share.streamlit.io)
- Sign in with GitHub
- Click "New app"
- Select repository and branch
- Set main file to `app.py`
- Click "Deploy"

3. **Access Application**
- Your app will be available at `https://your-username-app-name.streamlit.app`

### AWS EC2 Deployment

1. **Launch EC2 Instance**
   - Choose Ubuntu 20.04 LTS
   - Instance type: t2.medium (minimum)
   - Security group: Open port 8501

2. **Connect and Setup**
```bash
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3-pip python3-venv -y

# Clone repository
git clone your-repo-url
cd customer-churn-prediction

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Initialize database
python3 -c "from database import init_database; init_database()"
```

3. **Run Application**
```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

4. **Use Screen for Persistent Process**
```bash
screen -S churn-app
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
# Press Ctrl+A then D to detach
```

### Google Cloud Run

1. **Create `cloud-run.yaml`**
```yaml
runtime: python310
entrypoint: streamlit run app.py
env:
  - name: PORT
    value: "8501"
```

2. **Deploy**
```bash
gcloud run deploy churn-prediction \
  --source . \
  --platform managed \
  --region us-central1
```

### Heroku Deployment

1. **Create `Procfile`**
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

2. **Create `runtime.txt`**
```
python-3.10.0
```

3. **Deploy**
```bash
heroku login
heroku create your-app-name
git push heroku main
```

### Azure App Service

1. **Create Python App Service**
```bash
az webapp create \
  --resource-group myResourceGroup \
  --plan myAppServicePlan \
  --name myAppName \
  --runtime "PYTHON|3.10"
```

2. **Deploy Code**
```bash
az webapp up \
  --resource-group myResourceGroup \
  --name myAppName \
  --runtime "PYTHON|3.10"
```

---

## Production Checklist

### Before Deployment

- [ ] Update `requirements.txt` with exact versions
- [ ] Set `DEBUG=False` in environment
- [ ] Configure proper logging
- [ ] Set secure database location
- [ ] Update CORS settings if needed
- [ ] Configure HTTPS certificates
- [ ] Setup monitoring and alerts
- [ ] Create automated backups
- [ ] Test all features thoroughly
- [ ] Review security settings

### Environment Configuration

Create `.env` for production:
```
DEBUG=False
LOG_LEVEL=WARNING
DATABASE_PATH=/var/lib/churn/database.db
DATASETS_DIR=/var/lib/churn/datasets
MODELS_DIR=/var/lib/churn/models
LOGS_DIR=/var/log/churn
SESSION_TIMEOUT=3600
```

### Security Hardening

1. **Update Dependencies Regularly**
```bash
pip list --outdated
pip install --upgrade package_name
```

2. **Set File Permissions**
```bash
chmod 700 database/
chmod 700 models/
chmod 755 logs/
```

3. **Configure Firewall**
```bash
sudo ufw allow 8501/tcp
sudo ufw enable
```

4. **Enable SSL/TLS**
```bash
streamlit run app.py \
  --logger.level=info \
  --client.showErrorDetails=false
```

### Monitoring and Logging

1. **Setup Log Rotation**
```bash
# Create logrotate config
echo "/var/log/churn/*.log {
  daily
  rotate 7
  compress
  delaycompress
  notifempty
  create 0640 www-data www-data
  sharedscripts
}" | sudo tee /etc/logrotate.d/churn
```

2. **Monitor Resource Usage**
```bash
# Use tools like:
- Grafana for visualization
- Prometheus for metrics
- ELK stack for logging
```

3. **Setup Health Checks**
```bash
# Add cron job for periodic health checks
0 */6 * * * curl -f http://localhost:8501/_stcore/health || notify_admin
```

### Scaling

1. **Load Balancing**
   - Use Nginx as reverse proxy
   - Setup multiple app instances
   - Use session affinity

2. **Database Optimization**
   - Consider PostgreSQL for production
   - Setup database replication
   - Implement connection pooling

3. **Caching**
   - Implement Redis for session caching
   - Cache model predictions
   - Use CDN for static assets

### Backup Strategy

1. **Automated Backups**
```bash
#!/bin/bash
# Daily backup script
BACKUP_DIR="/backups/churn"
mkdir -p $BACKUP_DIR
tar -czf $BACKUP_DIR/backup_$(date +%Y%m%d).tar.gz \
  database/ datasets/ models/
```

2. **Backup Frequency**
   - Daily incremental backups
   - Weekly full backups
   - Monthly archive backups

---

## Troubleshooting Deployment

### Issue: Port 8501 Already in Use
```bash
# Find process using port
lsof -i :8501

# Kill process
kill -9 <PID>
```

### Issue: Database Locked
```bash
# Remove corrupted database
rm database/churn_prediction.db

# Reinitialize
python -c "from database import init_database; init_database()"
```

### Issue: Memory Issues
```bash
# Monitor memory usage
free -h

# Clear model cache if needed
rm -rf models/*.joblib
```

### Issue: Slow Predictions
- Optimize model complexity
- Implement prediction caching
- Use smaller feature set
- Consider GPU acceleration

---

## Performance Optimization

### Application Level
- Enable Streamlit caching decorators
- Use session state for expensive computations
- Lazy load large datasets

### Database Level
- Add indexes to frequently queried columns
- Partition large tables
- Implement connection pooling

### Model Level
- Use model quantization
- Implement batch prediction
- Cache model predictions

---

## Support and Documentation

For additional help:
- Check application logs in `logs/` directory
- Review README.md for usage guide
- Consult scikit-learn and Streamlit documentation
- Open issues on GitHub repository

---

**Last Updated:** 2024
**Version:** 1.0.0
