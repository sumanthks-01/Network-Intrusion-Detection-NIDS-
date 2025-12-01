// Authentication handling
document.getElementById('loginForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    
    try {
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ email, password })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            localStorage.setItem('userEmail', email);
            localStorage.setItem('isLoggedIn', 'true');
            window.location.href = '/dashboard';
        } else {
            alert(data.detail || 'Login failed');
        }
    } catch (error) {
        alert('Network error. Please try again.');
    }
});

document.getElementById('signupForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const email = document.getElementById('signupEmail').value;
    const password = document.getElementById('signupPassword').value;
    const confirmPassword = document.getElementById('confirmPassword').value;
    
    if (password !== confirmPassword) {
        alert('Passwords do not match');
        return;
    }
    
    try {
        const response = await fetch('/api/auth/signup', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ email, password })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            alert('Account created successfully! Please login.');
            hideSignup();
            document.getElementById('signupForm').reset();
        } else {
            alert(data.detail || 'Signup failed');
        }
    } catch (error) {
        alert('Network error. Please try again.');
    }
});

function showSignup() {
    document.getElementById('signupModal').style.display = 'block';
}

function hideSignup() {
    document.getElementById('signupModal').style.display = 'none';
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('signupModal');
    if (event.target === modal) {
        hideSignup();
    }
}

// Check if already logged in
if (localStorage.getItem('isLoggedIn') === 'true') {
    window.location.href = '/dashboard';
}