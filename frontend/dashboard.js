// Dashboard functionality
let detectionChart;
let isDetectionRunning = false;
let isMockRunning = false;
let benignCount = 0;
let attackCount = 0;
const API_BASE_URL = 'http://localhost:8000';
const mockAttacks = ['DoS Hulk', 'SSH-Patator', 'FTP-Patator', 'Web Attack XSS', 'PortScan', 'Bot', 'DDoS'];

// Check authentication
if (localStorage.getItem('isLoggedIn') !== 'true') {
    window.location.href = '/';
}

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    // Set user email
    document.getElementById('userEmail').textContent = localStorage.getItem('userEmail');
    
    // Initialize chart
    initChart();
    
    // Get network info
    getNetworkInfo();
    
    // Update status
    updateStatus();
});

function initChart() {
    const ctx = document.getElementById('detectionChart').getContext('2d');
    detectionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Benign Traffic',
                data: [],
                borderColor: '#28a745',
                backgroundColor: 'rgba(40, 167, 69, 0.1)',
                tension: 0.4
            }, {
                label: 'Attack Traffic',
                data: [],
                borderColor: '#dc3545',
                backgroundColor: 'rgba(220, 53, 69, 0.1)',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Count'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Time'
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            animation: {
                duration: 500
            }
        }
    });
}

function getNetworkInfo() {
    // Simulate network detection
    const networkTypes = ['WiFi', 'Ethernet', 'Mobile'];
    const interfaces = ['wlan0', 'eth0', 'ppp0'];
    
    const randomType = networkTypes[Math.floor(Math.random() * networkTypes.length)];
    const randomInterface = interfaces[Math.floor(Math.random() * interfaces.length)];
    
    document.getElementById('networkType').textContent = randomType;
    document.getElementById('networkInterface').textContent = randomInterface;
}

async function startDetection() {
    try {
        // Check backend connection
        const response = await fetch(`${API_BASE_URL}/api/health/`);
        if (!response.ok) throw new Error('Backend not available');
        
        isDetectionRunning = true;
        document.getElementById('startBtn').disabled = true;
        document.getElementById('stopBtn').disabled = false;
        document.getElementById('detectionStatus').textContent = 'Running';
        document.getElementById('detectionStatus').parentElement.className = 'status running';
        
        // Start real detection polling
        startRealDetection();
    } catch (error) {
        alert('Backend not available. Starting simulation mode.');
        isDetectionRunning = true;
        document.getElementById('startBtn').disabled = true;
        document.getElementById('stopBtn').disabled = false;
        document.getElementById('detectionStatus').textContent = 'Running (Simulation)';
        document.getElementById('detectionStatus').parentElement.className = 'status running';
        startSimulation();
    }
}

function stopDetection() {
    isDetectionRunning = false;
    isMockRunning = false;
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
    document.getElementById('detectionStatus').textContent = 'Stopped';
    document.getElementById('detectionStatus').parentElement.className = 'status stopped';
}

async function startRealDetection() {
    if (!isDetectionRunning) return;
    
    try {
        // Fetch recent detections from backend
        const response = await fetch(`${API_BASE_URL}/api/stats/detections?limit=10`);
        const data = await response.json();
        
        if (data.detections) {
            processDetections(data.detections);
        }
    } catch (error) {
        console.error('Error fetching detections:', error);
    }
    
    // Continue polling
    setTimeout(startRealDetection, 3000);
}

function startSimulation() {
    if (!isDetectionRunning && !isMockRunning) return;
    
    // Simulate detection data
    const now = new Date().toLocaleTimeString();
    const benignIncrease = Math.floor(Math.random() * 5) + 1;
    const attackIncrease = Math.random() < 0.3 ? Math.floor(Math.random() * 3) + 1 : 0;
    
    benignCount += benignIncrease;
    attackCount += attackIncrease;
    
    // Update chart
    updateChart(now, benignIncrease, attackIncrease);
    
    // Update counters
    document.getElementById('benignCount').textContent = benignCount;
    document.getElementById('attackCount').textContent = attackCount;
    
    // Add recent detection if attack
    if (attackIncrease > 0) {
        const attackType = mockAttacks[Math.floor(Math.random() * mockAttacks.length)];
        addRecentDetection(attackType, 'attack');
        
        // Save to database if mock is running
        if (isMockRunning) {
            saveMockDetection(attackType);
        }
    }
    
    // Continue simulation
    setTimeout(startSimulation, 2000);
}

function updateChart(time, benign, attack) {
    const maxDataPoints = 20;
    
    // Add new data
    detectionChart.data.labels.push(time);
    detectionChart.data.datasets[0].data.push(benign);
    detectionChart.data.datasets[1].data.push(attack);
    
    // Remove old data if too many points
    if (detectionChart.data.labels.length > maxDataPoints) {
        detectionChart.data.labels.shift();
        detectionChart.data.datasets[0].data.shift();
        detectionChart.data.datasets[1].data.shift();
    }
    
    detectionChart.update('none');
}

function updateStatus() {
    const statusElement = document.getElementById('detectionStatus').parentElement;
    if (isDetectionRunning) {
        statusElement.className = 'status running';
    } else {
        statusElement.className = 'status stopped';
    }
}

function logout() {
    localStorage.removeItem('userEmail');
    localStorage.removeItem('isLoggedIn');
    window.location.href = '/';
}

function processDetections(detections) {
    detections.forEach(detection => {
        if (detection.prediction !== 'BENIGN') {
            attackCount++;
            addRecentDetection(detection.prediction, 'attack');
        } else {
            benignCount++;
        }
    });
    
    // Update UI
    document.getElementById('benignCount').textContent = benignCount;
    document.getElementById('attackCount').textContent = attackCount;
    
    const now = new Date().toLocaleTimeString();
    updateChart(now, benignCount, attackCount);
}

function addRecentDetection(type, category) {
    const detectionList = document.getElementById('detectionList');
    const item = document.createElement('div');
    item.className = `detection-item ${category}`;
    item.textContent = `${new Date().toLocaleTimeString()} - ${type}`;
    
    detectionList.insertBefore(item, detectionList.firstChild);
    
    // Keep only last 10 items
    while (detectionList.children.length > 10) {
        detectionList.removeChild(detectionList.lastChild);
    }
}

async function saveMockDetection(attackType) {
    try {
        const mockData = {
            features: [Math.random(), Math.random(), Math.random()],
            meta: {
                attack_type: attackType,
                confidence: Math.random() * 0.3 + 0.7, // 70-100%
                src_ip: '192.168.1.100',
                dst_ip: '192.168.1.1',
                timestamp: new Date().toISOString()
            }
        };
        
        await fetch(`${API_BASE_URL}/api/detections/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(mockData)
        });
    } catch (error) {
        console.error('Failed to save mock detection:', error);
    }
}

function startMockDetection() {
    if (isMockRunning) return;
    
    isMockRunning = true;
    document.querySelector('.demo-btn').textContent = '🔄 Mock Detection Running...';
    document.querySelector('.demo-btn').disabled = true;
    
    // Run mock for 30 seconds
    startSimulation();
    
    setTimeout(() => {
        isMockRunning = false;
        document.querySelector('.demo-btn').textContent = '🔍 Wanna know how our system works?';
        document.querySelector('.demo-btn').disabled = false;
    }, 30000);
}

// Add some animation to stat cards
document.addEventListener('DOMContentLoaded', function() {
    const statCards = document.querySelectorAll('.stat-card');
    statCards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
        card.style.animation = 'slideIn 0.5s ease-out forwards';
    });
});