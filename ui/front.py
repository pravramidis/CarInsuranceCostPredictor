import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.core.window import Window
from kivymd.app import MDApp
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.pickers import MDDockedDatePicker
from kivy.metrics import dp

import eval

class DatePickerMixin:
    date_dialog = None

    def show_date_picker(self, widget, focus):
        if not focus:
            return
        
        self.trigger_widget = widget

        if not self.date_dialog:
            self.date_dialog = MDDockedDatePicker()
            self.date_dialog.bind(on_ok=self.on_ok, on_cancel=self.on_cancel)
        
        # Positioning the dialog based on the widget's properties
        self.date_dialog.pos = [
            widget.center_x - self.date_dialog.width / 2,
            widget.y - (self.date_dialog.height - dp(142))
        ]
        self.date_dialog.open()

    def on_ok(self, instance):
        value = instance.get_date()[0]
        if value:
            formatted_date = value.strftime('%m/%d/%Y')
            if hasattr(self, 'trigger_widget') and self.trigger_widget:
                self.trigger_widget.text = formatted_date
            self.date_dialog.dismiss()

    def on_cancel(self,instance):
        if self.date_dialog:
            self.date_dialog.dismiss()

class DropdownMenuMixin:
    def create_menu(self, textfield, menu_items):
        """Create and return a dropdown menu for the specified text field and menu items."""
        menu = MDDropdownMenu(
            caller=textfield,
            items=menu_items,
            width_mult=4,
            position='bottom'
        )
        return menu

    def open_menu(self, menu):
        """Open the specified menu."""
        if menu:
            menu.open()

    def set_item(self, text, menu, text_field):
        """Set the selected item's text to the TextField and dismiss the menu."""
        text_field.text = text
        menu.dismiss()

#Define different screens
class LoginScreen(Screen):
    def __init__(self, **kwargs):
        super(LoginScreen, self).__init__(**kwargs)
        
        fl = FloatLayout()
        bg = Image(source='login_background.png', allow_stretch=True, keep_ratio=False)
        fl.add_widget(bg)
        self.add_widget(fl)

class MainScreen(Screen, DatePickerMixin, DropdownMenuMixin):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
                
        
        fl = FloatLayout()
        bg = Image(source='main_background.png', allow_stretch=True, keep_ratio=False)
        fl.add_widget(bg)
        self.add_widget(fl)

        self.menu = None
   
    
    def on_leave(self):
        # Make sure to clean up by dismissing the menu if it's open
        if self.menu:
            self.menu.dismiss()
    
    def on_dropdown_focus(self, instance, value):
        area_field = self.ids.area_field
            
        menu_items = [
            {"text": "Rural", "on_release": lambda x="Rural": self.set_item(x, self.menu, area_field)},
            {"text": "Urban", "on_release": lambda x="Urban": self.set_item(x, self.menu, area_field)}
        ]
        
        self.menu = self.create_menu(area_field, menu_items)
        self.open_menu(self.menu) 
            

    def on_text_field_focus(self, instance, value):
        if value:  # if the text field is focused
            self.show_date_picker(instance,value)
    
    def date_of_birth(self):
        return self.ids.dob.text
    
    def licence_issue_date(self):
        return self.ids.licence_issue_date_field.text
    
    def area(self):
        return self.ids.area_field.text
    
    def seniority(self):
        return self.ids.seniority.text
    

class VehicleScreen(Screen, DropdownMenuMixin):
    def __init__(self, **kwargs):
        super(VehicleScreen, self).__init__(**kwargs)
                
        
        fl = FloatLayout()
        bg = Image(source='main_background.png', allow_stretch=True, keep_ratio=False)
        fl.add_widget(bg)
        self.add_widget(fl)

    def on_dropdown_focus(self, instance, value):
        if instance == self.ids.type_field:
            menu_items = [
            {"text": "Passenger Car", "on_release": lambda x="Passenger Car": self.set_item(x,self.menu, instance)},
            {"text": "Motorbike", "on_release": lambda x="Motorbike": self.set_item(x,self.menu, instance)},
            {"text": "Van", "on_release": lambda x="Van": self.set_item(x,self.menu, instance)},
            {"text": "Agricultural Vehicle", "on_release": lambda x="Agricultural Vehicle": self.set_item(x,self.menu, instance)}
            ]
        elif instance == self.ids.fuel_field:   
            menu_items = [
            {"text": "Petrol", "on_release": lambda x="Petrol": self.set_item(x,self.menu, instance)},
            {"text": "Diesel", "on_release": lambda x="Diesel": self.set_item(x,self.menu, instance)}
            ]
        self.menu = self.create_menu(instance, menu_items)
        self.open_menu(self.menu) 
    
    def get_vehicle_type(self):
        return self.ids.type_vehicle.text
    
    def get_fuel_type(self):
        return self.ids.fuel_field.text
    
    def get_registration_year(self):
        return self.ids.registration.text
    
    def get_horse_power(self):
        return self.ids.power.text
    
    def get_cylinder_capacity(self):
        return self.ids.cylinder.text
    
    def get_weight(self):
        return self.ids.weight.text
    
    def get_length(self):
        return self.ids.length.text
    
    def get_value(self):
        return self.ids.value.text
    

class InsuranceScreen(Screen, DatePickerMixin, DropdownMenuMixin):
    def __init__(self, **kwargs):
        super(InsuranceScreen, self).__init__(**kwargs)
                
        
        fl = FloatLayout()
        bg = Image(source='main_background.png', allow_stretch=True, keep_ratio=False)
        fl.add_widget(bg)
        self.add_widget(fl)

    def on_text_field_focus(self, instance, value):
        if value:  # if the text field is focused
            self.show_date_picker(instance,value)
    
    def on_dropdown_focus(self, instance, value):
        if instance == self.ids.driver:
            menu_items = [
            {"text": "Yes", "on_release": lambda x="Yes": self.set_item(x,self.menu, instance)},
            {"text": "No", "on_release": lambda x="No": self.set_item(x,self.menu, instance)}
            ]
        elif instance == self.ids.channel:   
            menu_items = [
            {"text": "Agent", "on_release": lambda x="Agent": self.set_item(x,self.menu, instance)},
            {"text": "Insurance Brokers", "on_release": lambda x="Insurance Brokers": self.set_item(x,self.menu, instance)}
            ]
        elif instance == self.ids.payment:
            menu_items = [
            {"text": "Half-yearly", "on_release": lambda x="Half-yearly": self.set_item(x,self.menu, instance)},
            {"text": "Annual", "on_release": lambda x="Annual": self.set_item(x,self.menu, instance)}
            ]
        self.menu = self.create_menu(instance, menu_items)
        self.open_menu(self.menu) 
    
    def get_start_contract(self):
        return self.ids.start_contract.text

    def get_last_renewal(self):
        return self.ids.last_renewal.text
    
    def get_next_renewal(self):
        return self.ids.next_renewal.text
    
    def get_driver(self):
        return self.ids.driver.text
    
    def get_payment(self):
        return self.ids.payment.text
    
    def get_channel(self):
        return self.ids.channel.text
    
    def get_claims(self):
        return self.ids.claims.text
    
    def get_policies(self):
        return self.ids.policies.text
    
    def get_lapse(self):
        return self.ids.lapse.text
    
    def get_ratio_claims(self):
        return self.ids.ratio_claims.text


class PriceScreen(Screen):
    def __init__(self, **kwargs):
        super(PriceScreen, self).__init__(**kwargs)
                
        self.data = None
        fl = FloatLayout()
        bg = Image(source='main_background.png', allow_stretch=True, keep_ratio=False)
        fl.add_widget(bg)
        self.add_widget(fl)

    

    def receive_data(self, data):
        self.data = data
        date_of_birth = self.data["main_screen"]["date_of_birth"]
        licence_issue_date = self.data["main_screen"]["licence_issue_date"]
        area = self.data["main_screen"]["area"]
    

class ScreenManager(ScreenManager):
    pass



class SafeWheels(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "Blue"
        return Builder.load_file('front.kv')
    
    def get_all_data(self):
        data = {}

        main_screen = self.root.get_screen("main")
        data["main_screen"] = {
            "date_of_birth": main_screen.date_of_birth(),
            "licence_issue_date": main_screen.licence_issue_date(),
            "area": main_screen.area(),
            "seniority": main_screen.seniority()
        }

        vehicle_screen = self.root.get_screen("vehicle")
        data["vehicle_screen"] = {
            "vehicle_type": vehicle_screen.get_vehicle_type(),
            "fuel_type": vehicle_screen.get_fuel_type(),
            "registration_year": vehicle_screen.get_registration_year(),
            "horse_power": vehicle_screen.get_horse_power(),
            "cylinder_capacity": vehicle_screen.get_cylinder_capacity(),
            "weight": vehicle_screen.get_weight(),
            "length": vehicle_screen.get_length(),
            "value": vehicle_screen.get_value()
        }

        insurance_screen = self.root.get_screen("insurance")
        data["insurance_screen"] = {
            "start_contract": insurance_screen.get_start_contract(),
            "last_renewal": insurance_screen.get_last_renewal(),
            "next_renewal": insurance_screen.get_next_renewal(),
            "drivers": insurance_screen.get_driver(),
            "payment": insurance_screen.get_payment(),
            "channel": insurance_screen.get_channel(),            
            "claims": insurance_screen.get_claims(),
            "policies": insurance_screen.get_policies(),
            "lapse": insurance_screen.get_lapse(),
            "ratio_claims": insurance_screen.get_ratio_claims()
        }

        price_screen = self.root.get_screen("price")
        price_screen.receive_data(data)

        # Navigate to the PriceScreen
        self.root.current = "price"


if __name__ == '__main__':
    SafeWheels().run()