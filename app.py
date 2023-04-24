import streamlit as st
from multiapp import MultiApp
import requests
import myapp # import your app modules here

app = MultiApp()

st.markdown("""
# Credit Card Fraud Detection
""")


def signup(email, password):
    try:
        # Send password reset request using Firebase Authentication API
        signUpUrl = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=AIzaSyBQhbmyyimlX-o81v77Tt57DFExIAgg4so"
        data = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }
        response = requests.post(signUpUrl, json=data)

        # Display success message
        st.success("Your account has been created.")
    except:
        # Display error message
        st.error("There was an error resetting your password. Please try again later.") 

    st.info('You can now login')    

def resetPass(email):
    try:
        # Send password reset request using Firebase Authentication API
        resetUrl = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key=AIzaSyBQhbmyyimlX-o81v77Tt57DFExIAgg4so"
        data = {
            "requestType": "PASSWORD_RESET",
            "email": email
        }
        response = requests.post(resetUrl, json=data)
        # Display success message
        st.success("Check your email address to find the reset password link.")
    except:
        # Display error message
        st.error("There was an error resetting your password. Please try again later.")  

def login(email, password):
    try:
        auth_url = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=AIzaSyBQhbmyyimlX-o81v77Tt57DFExIAgg4so"

        data = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }

        response = requests.post(auth_url, json=data)
        if response.ok:
            return True
        else:
            st.write("Authentication failed with error:", response.json()['error']['message'])
    except Exception:
        st.error("Error:")


@st.cache_resource
def create_user_session():
    return {'logged_in': False}

def main():

    user_session = create_user_session()

    # if user_session['logged_in'] is True:
        # app.add_app("Best model predictor", myapp.page1)
        # app.add_app("Build your own model", myapp.page2)
                
    #     # # The main app
        # app.run()
        


    # Authentication
    option = st.sidebar.selectbox('Login/Signup/Reset-Password', ['Login', 'Sign up', 'Reset Password'])

    # Sign up Block
    if option == 'Sign up':
        email = st.sidebar.text_input("Email")
        password = st.sidebar.text_input("Password", type="password")
        submit = st.sidebar.button('Create my account')
        if submit:
            signup(email, password)

    if option == "Reset Password":
        email = st.sidebar.text_input("Email")
        if st.sidebar.button("Reset Password"):
            resetPass(email)

    if option == "Login":
        email = st.sidebar.text_input("Email")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.checkbox("Login"):     
            if login(email, password):
                user_session['logged_in'] = True
                app.add_app("Best model predictor", myapp.page1)
                app.add_app("Build your own model", myapp.page2)
                
                # The main app
                app.run()

if __name__ == '__main__':
    main()

    
# https://firebase.google.com/docs/reference/rest/auth


# def main():

#     if user_session['logged_in'] is True:
#         st.write("Authentication already established")

#     # Authentication
#     option = st.sidebar.selectbox('Login/Signup/Reset-Password', ['Login', 'Sign up', 'Reset Password'])

#     # Sign up Block
#     if option == 'Sign up':
#         email = st.sidebar.text_input("Email")
#         password = st.sidebar.text_input("Password", type="password")
#         submit = st.sidebar.button('Create my account')
#         if submit:
#             signup(email, password)

#     if option == "Reset Password":
#         email = st.sidebar.text_input("Email")
#         if st.sidebar.button("Reset Password"):
#             resetPass(email)

#     if option == "Login":
#         user_session = create_user_session()
#         email = st.sidebar.text_input("Email")
#         password = st.sidebar.text_input("Password", type="password")
#         if st.sidebar.button("Login"):     
#             if login(email, password):
#                 user_session['logged_in'] = True
#                 app.add_app("Best model predictor", myapp.page1)
#                 app.add_app("Build your own model", myapp.page2)
                
#                 # The main app
#                 app.run()
#     # uploaded_file = st.file_uploader("Upload Files",type=['csv'])
#     # app.add_app("Best model predictor", myapp.page1)
#     # app.add_app("Build your own model", myapp.page2)
    
#     # # The main app
#     # app.run()